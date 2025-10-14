# Foldy Frontend

[![Frontend Tests](https://github.com/JBEI/foldy/actions/workflows/frontend-tests.yml/badge.svg)](https://github.com/JBEI/foldy/actions/workflows/frontend-tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/JBEI/foldy/badge.svg?branch=main)](https://coveralls.io/github/JBEI/foldy?branch=main)

Folding service frontend SPA (Single Page Application).

## Testing

The project uses Jest and React Testing Library for testing. Tests are co-located with their respective components/modules.

### Running Tests

```bash
# Run tests in watch mode (development)
npm test

# Run tests with coverage report
npm run test:coverage

# Run tests in CI mode (used by GitHub Actions)
npm run test:ci

# Run tests for a specific file
npm run test:file src/api/foldApi.test.tsx
```

### Code Quality

```bash
# Run ESLint
npm run lint

# Fix ESLint issues automatically where possible
npm run lint:fix
```

### Test Organization

- **Unit Tests**: Tests for utility functions, API modules, and hooks
- **Component Tests**: Tests for React components

### Writing Tests

1. **For API modules**:
   - Mock axios using Jest's mocking system
   - Test success and error cases
   - Example: `src/api/foldApi.test.tsx`

2. **For Utility Functions**:
   - Test with different inputs and edge cases
   - Example: `src/util/boltzYamlHelper.test.tsx`

3. **For Components**:
   - Use React Testing Library to render and interact with components
   - Test user interactions using `fireEvent` or `userEvent`
   - Example: `src/components/FoldView/DockTab.test.tsx`

### CI Integration

Tests are automatically run on GitHub Actions when:
- Pushing to main/master branch
- Creating a pull request to main/master branch
- Only runs when changes are made in the frontend directory
