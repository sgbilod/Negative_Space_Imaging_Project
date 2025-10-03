# Contributing to Negative Space Imaging System
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

Thank you for considering contributing to the Negative Space Imaging System! This document outlines our contribution guidelines and processes to help you get started.

## Code of Conduct

By participating in this project, you agree to uphold our Code of Conduct. Please report unacceptable behavior to conduct@negativespacesystems.com.

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report:

1. Check the [issue tracker](https://github.com/yourusername/negative-space-imaging/issues) to see if the issue has already been reported
2. If you're unable to find an open issue addressing the problem, create a new one

When creating a bug report, include as much detail as possible:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Screenshots if applicable
- Your environment (OS, browser, etc.)
- Any additional context

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

1. Use a clear and descriptive title
2. Provide a detailed description of the suggested enhancement
3. Explain why this enhancement would be useful
4. Include mockups or examples if applicable

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`npm run lint && npm test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Process

### Setting Up Development Environment

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/negative-space-imaging.git
   cd negative-space-imaging
   ```

2. Install dependencies
   ```bash
   npm install
   ```

3. Configure environment variables
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. Start development server
   ```bash
   npm run dev
   ```

### Code Style

- Follow the existing code style
- Use TypeScript for type safety
- Document your code with JSDoc comments
- Write comprehensive tests

We use ESLint and Prettier to enforce coding standards:

```bash
# Check code style
npm run lint

# Automatically fix style issues
npm run lint:fix

# Format code
npm run format
```

### Testing

Write tests for all new features and bug fixes:

```bash
# Run all tests
npm test

# Run tests with coverage
npm run test:coverage
```

Aim for 100% test coverage for new code.

### Commits

We follow [Conventional Commits](https://www.conventionalcommits.org/) for our commit messages:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code changes that neither fix bugs nor add features
- `perf`: Performance improvements
- `test`: Adding or correcting tests
- `chore`: Changes to the build process or auxiliary tools

### Branch Naming Convention

- `feature/` - New features
- `bugfix/` - Bug fixes
- `hotfix/` - Critical fixes for production
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

## HIPAA Compliance

When contributing to this project, be aware that it's designed to be HIPAA compliant for medical applications:

- Never commit real patient data or PHI
- Use synthetic data for testing
- Be cautious with logging to avoid exposing sensitive information
- Follow security best practices

## Licensing

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

If you have questions about contributing, please open an issue or contact the maintainers at contributors@negativespacesystems.com.

Thank you for contributing to the Negative Space Imaging System!
