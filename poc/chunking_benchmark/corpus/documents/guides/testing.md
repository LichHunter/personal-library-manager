# Testing Guide

How we test CloudFlow.

## Test Types

### Unit Tests

Test individual functions and methods.

Location: `*_test.go` files alongside source

```bash
# Run all unit tests
make test-unit

# Run specific package
go test ./api/handlers/...

# With coverage
go test -cover ./...
```

### Integration Tests

Test service interactions.

Location: `tests/integration/`

```bash
# Requires local environment running
make test-integration
```

### End-to-End Tests

Test full user flows.

Location: `tests/e2e/`

```bash
# Uses Playwright
make test-e2e
```

## Writing Tests

### Unit Test Example

```go
func TestWorkflowValidation(t *testing.T) {
    tests := []struct {
        name    string
        input   Workflow
        wantErr bool
    }{
        {
            name: "valid workflow",
            input: Workflow{
                Name: "test",
                Steps: []Step{{ID: "step1"}},
            },
            wantErr: false,
        },
        {
            name: "missing name",
            input: Workflow{
                Steps: []Step{{ID: "step1"}},
            },
            wantErr: true,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := tt.input.Validate()
            if (err != nil) != tt.wantErr {
                t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}
```

### Integration Test Example

```go
func TestCreateWorkflowAPI(t *testing.T) {
    // Setup
    db := setupTestDB(t)
    defer db.Close()
    
    server := setupTestServer(db)
    
    // Test
    resp, err := server.Client().Post(
        "/v1/workflows",
        "application/json",
        strings.NewReader(`{"name": "test"}`),
    )
    
    // Assert
    require.NoError(t, err)
    assert.Equal(t, 201, resp.StatusCode)
}
```

## Test Data

### Fixtures

Test fixtures are in `tests/fixtures/`:

- `workflows/` - Sample workflow definitions
- `users/` - Test user data
- `executions/` - Sample execution data

### Factories

Use factories for dynamic test data:

```go
user := factory.NewUser().
    WithRole("admin").
    Build()
```

## Coverage

We aim for:
- Unit tests: >80% coverage
- Integration tests: Critical paths covered
- E2E tests: Main user flows covered

Check coverage:

```bash
make coverage
# Opens coverage report in browser
```

## CI Pipeline

Tests run on every PR:

1. Lint
2. Unit tests
3. Integration tests
4. E2E tests (on main branch only)

All tests must pass before merge.

## Mocking

Use interfaces for dependencies, mock in tests:

```go
type WorkflowStore interface {
    Get(id string) (*Workflow, error)
    Create(w *Workflow) error
}

// In tests
type mockStore struct {
    workflows map[string]*Workflow
}

func (m *mockStore) Get(id string) (*Workflow, error) {
    return m.workflows[id], nil
}
```
