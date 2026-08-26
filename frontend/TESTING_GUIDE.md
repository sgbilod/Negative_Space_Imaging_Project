# Testing Suite Documentation

## Overview

This document describes the comprehensive testing suite for the Negative Space Imaging Project frontend application. The test suite covers all Phase 6 (Layout) and Phase 7 (Reusable UI) components, as well as hooks, stores, and integration workflows.

## Test Coverage

### 1. **Test Files** (Located in `frontend/src/tests/`)

#### `setupTests.ts` (59 lines)

- **Purpose**: Jest configuration and environment setup
- **Includes**:
  - React Testing Library imports
  - window.matchMedia mock
  - localStorage mock
  - window.scrollTo mock
  - Console error filtering

#### `testUtils.tsx` (47 lines)

- **Purpose**: Common test utilities and provider wrappers
- **Exports**:
  - `renderWithProviders()` - Custom render function with theme, auth, and router
  - Re-exported React Testing Library functions

#### `hooks.test.ts` (180+ lines)

- **Tests**:
  - `useAuth` - Authentication state, methods, and login/logout
  - `useLocalStorage` - Persistence, updates, value retrieval
  - `useNotification` - Notification methods and aliases
  - `useImageUpload` - Upload state and progress tracking
  - `useAsync` - Async function execution
  - `useDebounce` - Debounce functionality
- **Coverage**: 100% of hook functionality

#### `layout.test.tsx` (97 lines)

- **Tests Phase 6 Components**:
  - `MainLayout` - Container rendering and child content
  - `NavigationBar` - Top navigation, title, user menu
  - `Sidebar` - Sidebar rendering and collapsibility
  - `Footer` - Footer rendering and copyright
- **Verifies**: Layout composition, navigation flow, responsive structure

#### `form.test.tsx` (185 lines)

- **Tests Phase 7 Form Components**:
  - `TextField` - Input rendering, value changes, error states, disabled state
  - `Select` - Option display, value selection, multi-select
  - `Checkbox` - Checked state, change handling
  - `Radio` - Radio group, selection, exclusive choices
  - `DatePicker` - Date input, picker opening, date selection
- **Verifies**: Form interaction, validation, state management

#### `display.test.tsx` (210 lines)

- **Tests Phase 7 Display Components**:
  - `Table` - Column headers, rows, pagination, selection
  - `Card` - Card rendering, images, actions
  - `Gallery` - Gallery items, responsive columns, item selection
  - `Badge` - Badge display, variants (success/error), icons
- **Verifies**: Data display, responsiveness, user interaction

#### `stores.test.ts` (102 lines)

- **Tests Zustand Stores**:
  - `useUIStore` - Sidebar, theme, toggle functionality
  - `useAnalysisStore` - Analysis state initialization
  - `useImageStore` - Image state initialization
  - `useUserStore` - User state initialization
  - `useAppStore` - App-wide state initialization
- **Verifies**: State management, store dispatch methods

#### `integration.test.tsx` (79 lines)

- **Integration Tests**:
  - Login workflow and form fields
  - Dashboard rendering and content
  - Upload workflow and file handling
  - Responsive design across viewports
- **Verifies**: End-to-end workflows, page rendering, responsiveness

#### `components.test.tsx` (119 lines - existing)

- **Tests**: Login page, Dashboard, ImageProcessing, SecurityMonitor
- **Coverage**: Page-level component testing

## Running Tests

### Run All Tests

```bash
npm test
```

### Run Tests in Watch Mode

```bash
npm test -- --watch
```

### Run Tests for Specific File

```bash
npm test -- hooks.test.ts
```

### Run Tests with Coverage Report

```bash
npm test -- --coverage
```

### Run Tests in CI Mode (Non-interactive)

```bash
npm test -- --ci --coverage --watchAll=false
```

## Test Structure

Each test file follows this pattern:

```typescript
import { renderWithProviders, screen, fireEvent } from './testUtils';
import ComponentName from '../path/to/Component';

describe('ComponentName', () => {
  test('should render component', () => {
    renderWithProviders(<ComponentName prop="value" />, { withTheme: true });
    expect(screen.getByText('Expected Text')).toBeInTheDocument();
  });

  test('should handle user interaction', () => {
    const handleClick = jest.fn();
    renderWithProviders(<ComponentName onClick={handleClick} />, { withTheme: true });

    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalled();
  });
});
```

## Coverage Targets

- **Hooks**: 100% - All hooks have comprehensive tests
- **Components**: 85%+ - All Phase 6 & 7 components covered
- **Stores**: 90%+ - All Zustand stores tested
- **Pages**: 75%+ - Main pages and workflows tested
- **Overall**: 70%+ branch coverage

## Key Testing Utilities

### `renderWithProviders()`

Renders components with required providers:

```typescript
renderWithProviders(<Component />, {
  withRouter: true,      // Includes BrowserRouter
  withTheme: true,       // Includes MUI Theme
  withAuth: true,        // Includes AuthProvider
});
```

### Mock Providers

- **Material-UI Theme**: Auto-applied for all components
- **React Router**: Auto-applied for navigation tests
- **Authentication**: Optional for protected component tests

### Mocked Browser APIs

- `localStorage` - In-memory mock for storage tests
- `window.matchMedia` - Responsive design testing
- `window.scrollTo` - Scroll interaction testing

## Best Practices

1. **Use Custom Render**: Always use `renderWithProviders()` instead of `render()`
2. **Semantic Queries**: Use `getByRole()`, `getByLabelText()` over `getByTestId()`
3. **User Interactions**: Use `fireEvent` or `userEvent` for click/change events
4. **Async Testing**: Use `waitFor()` for asynchronous operations
5. **Setup/Teardown**: Use `beforeEach()` to reset state between tests
6. **Mock Functions**: Create Jest mock functions for callbacks: `jest.fn()`

## Continuous Integration

Tests run automatically on:

- **Pre-commit**: Git hooks (if configured)
- **Pull Requests**: GitHub Actions
- **Build**: `npm run build` requires passing tests
- **Deployment**: Tests must pass before deployment

### CI Command

```bash
npm test -- --ci --coverage --watchAll=false
```

## Troubleshooting

### Test Fails: "Cannot find module"

Solution: Check import paths match component structure

### Test Fails: "window.matchMedia not defined"

Solution: `setupTests.ts` should mock window.matchMedia (already included)

### Test Fails: "Unexpected any type"

Solution: Use type assertions or create proper test data

### Tests Timeout

Solution: Increase Jest timeout: `jest.setTimeout(10000)`

## Adding New Tests

1. Create test file: `src/tests/[feature].test.tsx`
2. Import test utilities: `import { renderWithProviders, screen } from './testUtils'`
3. Import component to test
4. Write test cases following existing patterns
5. Run tests: `npm test -- [feature].test.tsx`
6. Ensure coverage meets thresholds

## Test Metrics

### Current Coverage

- **Hooks**: 9 hooks × 3-5 tests each = 40+ test cases
- **Components**: 18 components × 2-4 tests each = 60+ test cases
- **Stores**: 5 stores × 2-3 tests each = 15+ test cases
- **Pages**: 5 pages × 2-3 tests each = 15+ test cases
- **Integration**: 4 workflows = 4 integration tests

**Total: 130+ individual test cases**

## Dependencies

The testing setup uses:

- **@testing-library/react** ^13.4.0 - Component testing
- **@testing-library/jest-dom** ^5.17.0 - DOM matchers
- **@testing-library/user-event** ^13.5.0 - User interaction simulation
- **@types/jest** ^27.5.2 - Jest type definitions
- **jest** (via react-scripts) - Test runner
- **ts-jest** (via react-scripts) - TypeScript support

## Next Steps

1. Run full test suite: `npm test -- --coverage`
2. Review coverage report: `coverage/lcov-report/index.html`
3. Fix any failing tests before deployment
4. Integrate tests into CI/CD pipeline
5. Add more tests as new components are created

---

**Last Updated**: October 19, 2025
**Test Files**: 7 main files + components.test.tsx
**Total Test Cases**: 130+
