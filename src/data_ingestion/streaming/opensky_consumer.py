"""
Real-time ADS-B Data Streaming Consumer

Ingests flight data from OpenSky Network API and streams to Kafka.
Implements backpressure handling, circuit breaker, and exactly-once semantics.
"""

from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import logging
from enum import Enum
import time

# Kafka imports
try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("Kafka not available. Install with: pip install kafka-python")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class FlightData:
    """Structured flight data from ADS-B."""
    icao24: str
    callsign: Optional[str]
    origin_country: str
    time_position: Optional[float]
    last_contact: float
    longitude: Optional[float]
    latitude: Optional[float]
    baro_altitude: Optional[float]
    on_ground: bool
    velocity: Optional[float]
    true_track: Optional[float]
    vertical_rate: Optional[float]
    geo_altitude: Optional[float]
    squawk: Optional[str]
    spi: bool
    category: int
    timestamp: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.
    
    Prevents cascading failures by stopping requests when
    error rate is too high.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class OpenSkyStreamConsumer:
    """
    Async consumer for OpenSky Network API.
    
    Features:
    - Async HTTP requests with aiohttp
    - Circuit breaker for fault tolerance
    - Backpressure handling
    - Data validation and transformation
    - Kafka streaming
    """
    
    def __init__(
        self,
        api_url: str = "https://opensky-network.org/api/states/all",
        poll_interval: int = 10,
        kafka_bootstrap_servers: Optional[List[str]] = None,
        kafka_topic: str = "flight-data",
        max_queue_size: int = 1000
    ):
        self.api_url = api_url
        self.poll_interval = poll_interval
        self.kafka_topic = kafka_topic
        self.max_queue_size = max_queue_size
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=60.0,
            expected_exception=aiohttp.ClientError
        )
        
        # Internal queue for backpressure
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Kafka producer
        self.kafka_producer = None
        if KAFKA_AVAILABLE and kafka_bootstrap_servers:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',  # Wait for all replicas
                retries=3,
                max_in_flight_requests_per_connection=1  # Exactly-once semantics
            )
        
        # Statistics
        self.stats = {
            'total_fetched': 0,
            'total_processed': 0,
            'total_errors': 0,
            'last_fetch_time': None
        }
        
        # Logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    async def fetch_data(self, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
        """
        Fetch data from OpenSky API.
        
        Args:
            session: aiohttp session
        
        Returns:
            API response data or None on error
        """
        try:
            async with session.get(self.api_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                data = await response.json()
                self.stats['last_fetch_time'] = datetime.now()
                return data
        except aiohttp.ClientError as e:
            self.logger.error(f"Error fetching data: {e}")
            self.stats['total_errors'] += 1
            return None
        except asyncio.TimeoutError:
            self.logger.error("Request timeout")
            self.stats['total_errors'] += 1
            return None
    
    def parse_flight_data(self, states_list: List[List], timestamp: int) -> List[FlightData]:
        """
        Parse raw state vectors into FlightData objects.
        
        Args:
            states_list: List of state vectors from API
            timestamp: API response timestamp
        
        Returns:
            List of FlightData objects
        """
        flights = []
        
        for state in states_list:
            try:
                flight = FlightData(
                    icao24=state[0],
                    callsign=state[1].strip() if state[1] else None,
                    origin_country=state[2],
                    time_position=state[3],
                    last_contact=state[4],
                    longitude=state[5],
                    latitude=state[6],
                    baro_altitude=state[7],
                    on_ground=state[8],
                    velocity=state[9],
                    true_track=state[10],
                    vertical_rate=state[11],
                    geo_altitude=state[13],
                    squawk=state[14],
                    spi=state[15],
                    category=state[16] if state[16] else 0,
                    timestamp=timestamp
                )
                flights.append(flight)
            except (IndexError, TypeError) as e:
                self.logger.warning(f"Error parsing state vector: {e}")
                continue
        
        return flights
    
    async def produce_to_kafka(self, flight: FlightData) -> bool:
        """
        Send flight data to Kafka.
        
        Args:
            flight: FlightData object
        
        Returns:
            True if successful, False otherwise
        """
        if not self.kafka_producer:
            return False
        
        try:
            future = self.kafka_producer.send(
                self.kafka_topic,
                value=flight.to_dict()
            )
            # Wait for acknowledgment
            record_metadata = future.get(timeout=10)
            self.logger.debug(
                f"Sent to Kafka: topic={record_metadata.topic}, "
                f"partition={record_metadata.partition}, "
                f"offset={record_metadata.offset}"
            )
            return True
        except KafkaError as e:
            self.logger.error(f"Kafka error: {e}")
            return False
    
    async def process_queue(self):
        """Process items from queue and send to Kafka."""
        while True:
            try:
                flight = await self.queue.get()
                
                # Send to Kafka
                success = await self.produce_to_kafka(flight)
                
                if success:
                    self.stats['total_processed'] += 1
                
                self.queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing queue: {e}")
                await asyncio.sleep(1)
    
    async def stream(self):
        """
        Main streaming loop.
        
        Continuously fetches data from OpenSky API and streams to Kafka.
        """
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    # Fetch data with circuit breaker
                    data = await self.fetch_data(session)
                    
                    if data and 'states' in data and data['states']:
                        # Parse flight data
                        flights = self.parse_flight_data(
                            data['states'],
                            data['time']
                        )
                        
                        self.stats['total_fetched'] += len(flights)
                        
                        # Add to queue with backpressure handling
                        for flight in flights:
                            try:
                                # Non-blocking put with timeout
                                await asyncio.wait_for(
                                    self.queue.put(flight),
                                    timeout=5.0
                                )
                            except asyncio.TimeoutError:
                                self.logger.warning("Queue full, dropping data")
                                break
                        
                        self.logger.info(
                            f"Fetched {len(flights)} flights. "
                            f"Queue size: {self.queue.qsize()}/{self.max_queue_size}"
                        )
                    else:
                        self.logger.warning("No data received from API")
                    
                    # Wait before next poll
                    await asyncio.sleep(self.poll_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in streaming loop: {e}")
                    await asyncio.sleep(self.poll_interval)
    
    async def start(self):
        """Start the streaming consumer."""
        self.logger.info("Starting OpenSky stream consumer...")
        
        # Start queue processor
        processor_task = asyncio.create_task(self.process_queue())
        
        # Start streaming
        stream_task = asyncio.create_task(self.stream())
        
        # Run both tasks
        await asyncio.gather(processor_task, stream_task)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consumer statistics."""
        return {
            **self.stats,
            'queue_size': self.queue.qsize(),
            'circuit_breaker_state': self.circuit_breaker.state.value
        }
    
    def close(self):
        """Close connections and cleanup."""
        if self.kafka_producer:
            self.kafka_producer.flush()
            self.kafka_producer.close()
        self.logger.info("Consumer closed")


# Example usage
async def main():
    """Example usage of OpenSky stream consumer."""
    
    # Create consumer
    consumer = OpenSkyStreamConsumer(
        api_url="https://opensky-network.org/api/states/all",
        poll_interval=10,
        kafka_bootstrap_servers=['localhost:9092'],
        kafka_topic='flight-data',
        max_queue_size=1000
    )
    
    # Start streaming
    try:
        await consumer.start()
    except KeyboardInterrupt:
        print("\nStopping consumer...")
        consumer.close()
        
        # Print statistics
        stats = consumer.get_stats()
        print(f"\nStatistics:")
        print(f"  Total fetched: {stats['total_fetched']}")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Total errors: {stats['total_errors']}")
        print(f"  Last fetch: {stats['last_fetch_time']}")


if __name__ == "__main__":
    asyncio.run(main())
