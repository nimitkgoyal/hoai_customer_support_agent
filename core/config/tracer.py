import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from core.config.logger import logger

try:
    # 1. Launch a completely fresh, in-memory Phoenix instance
    session = px.launch_app()
    logger.info(f"Phoenix Observability | Local dashboard launched. View traces at: {session.url}")
    
    # 2. Wire standard OpenTelemetry to send spans directly to port 6006
    endpoint = "http://localhost:6006/v1/traces"
    provider = TracerProvider()
    processor = SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    provider.add_span_processor(processor)
    
    # 3. Let LangChain's built-in hooks handle all trace mapping automatically
    LangChainInstrumentor().instrument(tracer_provider=provider)
    logger.info("Phoenix Observability | LangChain auto-tracing instrumentation activated safely.")
    
except Exception as e:
    logger.error(f"Phoenix Observability | Initialization encountered an error: {str(e)}")
