"""Local Chroma telemetry overrides for this app."""

from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpProductTelemetryClient(ProductTelemetryClient):
    """Disable Chroma product telemetry in local app deployments."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        del event
