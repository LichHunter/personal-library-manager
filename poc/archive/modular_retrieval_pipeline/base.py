"""Base pipeline abstractions for modular retrieval pipeline.

This module implements Unix pipe philosophy for retrieval pipelines:
- Components are pure functions (process() method)
- Data flows sequentially: output of component N → input of component N+1
- Components return new immutable objects (never modify input)
- Fail-fast error propagation (first error stops pipeline)
- Type validation between components

Example:
    >>> class UpperCase(Component):
    ...     def process(self, data: str) -> str:
    ...         return data.upper()
    ...
    >>> class AddExclamation(Component):
    ...     def process(self, data: str) -> str:
    ...         return data + "!"
    ...
    >>> pipeline = Pipeline().add(UpperCase()).add(AddExclamation())
    >>> result = pipeline.run("hello")
    >>> print(result)  # "HELLO!"
"""

from typing import Any, Protocol, TypeVar, Generic, runtime_checkable


# Type variables for input/output types
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@runtime_checkable
class Component(Protocol[InputT, OutputT]):
    """Protocol for pipeline components.

    Components must implement a process() method that takes input data
    and returns transformed output data. Components should be pure functions:
    - No side effects
    - No state mutation
    - Return new immutable objects

    The Unix pipe philosophy:
        component1.process(data) → data2
        component2.process(data2) → data3
        component3.process(data3) → result
    """

    def process(self, data: InputT) -> OutputT:
        """Process input data and return transformed output.

        Args:
            data: Input data (immutable)

        Returns:
            Transformed output data (new immutable object)

        Raises:
            Any exception will stop pipeline execution (fail-fast)
        """
        ...


class PipelineError(Exception):
    """Base exception for pipeline errors."""

    def __init__(
        self,
        message: str,
        component_index: int,
        component: Any,
        original_error: Exception | None = None,
    ):
        """Initialize pipeline error.

        Args:
            message: Error message
            component_index: Index of component that failed (0-based)
            component: The component that failed
            original_error: Original exception that caused the failure
        """
        self.component_index = component_index
        self.component = component
        self.original_error = original_error
        super().__init__(message)


class TypeValidationError(PipelineError):
    """Raised when type validation between components fails."""

    pass


class Pipeline(Generic[InputT, OutputT]):
    """Pipeline for chaining components with Unix pipe data flow.

    The pipeline chains components sequentially, where the output of each
    component becomes the input to the next component. This follows the
    Unix pipe philosophy: simple, composable, sequential data flow.

    Features:
    - Fluent API: Pipeline().add(c1).add(c2).add(c3)
    - Fail-fast: First error stops execution
    - Type validation: Ensures output type matches next input type
    - Immutable: Components cannot be modified after pipeline is built

    Example:
        >>> pipeline = (Pipeline()
        ...     .add(QueryRewriter())     # Query → RewrittenQuery
        ...     .add(QueryExpander())     # RewrittenQuery → ExpandedQuery
        ...     .add(BM25Scorer()))       # ExpandedQuery → ScoredChunks
        >>> result = pipeline.run(Query("test"))
    """

    def __init__(self):
        """Initialize empty pipeline."""
        self._components: list[Component] = []
        self._built = False

    def add(self, component: Component) -> "Pipeline":
        """Add a component to the pipeline.

        Components are executed in the order they are added.
        The output of component N becomes the input to component N+1.

        Args:
            component: Component to add (must implement Component protocol)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If pipeline is already built
            TypeError: If component doesn't implement Component protocol
        """
        if self._built:
            raise ValueError("Cannot add components to a built pipeline")

        # Validate component implements protocol
        if not isinstance(component, Component):
            raise TypeError(
                f"Component must implement Component protocol with process() method. "
                f"Got: {type(component).__name__}"
            )

        self._components.append(component)
        return self

    def build(self) -> "Pipeline":
        """Build the pipeline (makes it immutable).

        After building, no more components can be added.
        This is optional - run() will automatically build if needed.

        Returns:
            Self for method chaining
        """
        self._built = True
        return self

    def run(self, data: InputT) -> Any:
        """Run the pipeline on input data.

        Executes components sequentially:
        1. data → component[0].process(data) → data1
        2. data1 → component[1].process(data1) → data2
        3. data2 → component[2].process(data2) → result

        Fail-fast: If any component raises an exception, the pipeline
        stops immediately and wraps the error in a PipelineError.

        Args:
            data: Input data for the first component

        Returns:
            Output from the last component

        Raises:
            PipelineError: If any component fails
            ValueError: If pipeline has no components
        """
        if not self._components:
            raise ValueError("Cannot run empty pipeline. Add components with add()")

        # Auto-build if not already built
        if not self._built:
            self.build()

        # Unix pipe data flow: output of N → input of N+1
        current_data: Any = data

        for i, component in enumerate(self._components):
            try:
                # Type validation: Check if component can process current data
                # This is a runtime check since Python's type system is gradual
                current_data = component.process(current_data)

            except TypeError as e:
                # Type mismatch between components
                raise TypeValidationError(
                    f"Type validation failed at component {i} ({type(component).__name__}): "
                    f"Expected input type compatible with {type(current_data).__name__}, "
                    f"but component raised TypeError: {e}",
                    component_index=i,
                    component=component,
                    original_error=e,
                ) from e

            except Exception as e:
                # Any other error - fail fast
                raise PipelineError(
                    f"Component {i} ({type(component).__name__}) failed: {e}",
                    component_index=i,
                    component=component,
                    original_error=e,
                ) from e

        return current_data

    def __len__(self) -> int:
        """Return number of components in pipeline."""
        return len(self._components)

    def __repr__(self) -> str:
        """Return string representation of pipeline."""
        component_names = [type(c).__name__ for c in self._components]
        status = "built" if self._built else "building"
        return f"Pipeline({status}, components={component_names})"
