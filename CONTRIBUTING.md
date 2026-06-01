# Contributing Guidelines

Thank you for your interest in contributing to the Negative Space Imaging Project!

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and constructive
- Welcome diverse perspectives
- Focus on what's best for the project
- Report unacceptable behavior to maintainers

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Negative_Space_Imaging_Project.git`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Follow the setup instructions in `DEVELOPMENT.md`

## Development Workflow

### Before Starting

- Check existing issues and PRs to avoid duplicate work
- Discuss significant changes in an issue first
- Follow the project's coding standards

### Making Changes

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes following code standards
# Test your changes thoroughly
npm run test

# Ensure code quality
npm run lint
npm run format

# Commit with clear messages
git commit -m "fix: description of change

Optional longer description explaining why and how."
```

### Commit Messages

Follow conventional commits:

```
type(scope): subject

body (optional)
footer (optional, e.g. Fixes #123)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat(auth): add JWT refresh token rotation
fix(api): handle null image uploads
docs(setup): clarify Docker installation
test(analyzer): add negative space detection tests
```

## Pull Request Process

### Before Submitting

1. **Update your branch:** `git pull origin main`
2. **Run tests:** `npm run test`
3. **Check linting:** `npm run lint`
4. **Test locally:** Verify all functionality works
5. **Update docs:** If behavior changed, update documentation

### Creating the PR

1. Use the provided PR template
2. Give a clear, descriptive title
3. Explain the problem and solution
4. Reference related issues: `Fixes #123`
5. Include testing instructions
6. Request reviewers

### PR Title Format

```
[type] Brief description

Examples:
[feat] Add batch image processing
[fix] Correct memory leak in analyzer
[docs] Update deployment guide
[test] Add integration tests for API
```

### During Review

- Respond to feedback professionally
- Make requested changes in new commits
- Re-request review after updates
- Be patient with the review process

## Code Standards

### TypeScript/Node.js

```typescript
// Use strict types
function analyze(image: Buffer): Promise<AnalysisResult> {
  // implementation
}

// Add error handling
try {
  const result = await process(data);
} catch (error) {
  logger.error('Processing failed:', error);
  throw new ApiError('Processing failed', 500);
}

// Use const/let, not var
const config = loadConfig();

// Use arrow functions
const map = (arr: number[]) => arr.map(x => x * 2);

// Document complex functions
/**
 * Analyzes image for negative space patterns
 * @param image - Image buffer to analyze
 * @param options - Configuration options
 * @returns Promise resolving to analysis results
 */
```

### Python

```python
# Use type hints
def analyze_image(image: np.ndarray) -> Dict[str, Any]:
    """Analyze image for negative space patterns."""
    # implementation
    pass

# Follow PEP 8
class ImageAnalyzer:
    """Main analyzer class."""
    
    def __init__(self, config: Dict) -> None:
        self.config = config
    
    def process(self, image: np.ndarray) -> Dict:
        """Process image and return results."""
        pass

# Use meaningful variable names
negative_space_regions = detect_regions(image)

# Add docstrings
def detect_regions(image: np.ndarray) -> List[Region]:
    """
    Detect negative space regions in image.
    
    Args:
        image: Input image array
        
    Returns:
        List of detected regions
        
    Raises:
        ValueError: If image format invalid
    """
```

### React Components

```typescript
// Use functional components
interface ImageViewerProps {
  imageUrl: string;
  onAnalyze?: (result: AnalysisResult) => void;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  imageUrl,
  onAnalyze,
}) => {
  const [loading, setLoading] = useState(false);
  
  return (
    <div className="image-viewer">
      <img src={imageUrl} alt="Analysis target" />
    </div>
  );
};

// Use hooks for state
function useImageAnalysis(imageId: string) {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<Error | null>(null);
  
  return { result, error };
}
```

## Testing Requirements

### Coverage Targets

- Overall: 70%+
- Critical paths: 90%+
- New features: 80%+

### Writing Tests

```typescript
describe('ImageAnalyzer', () => {
  it('should detect negative space regions', async () => {
    const analyzer = new ImageAnalyzer();
    const result = await analyzer.analyze(testImage);
    expect(result.regions).toHaveLength(3);
  });
  
  it('should handle invalid images', async () => {
    const analyzer = new ImageAnalyzer();
    await expect(analyzer.analyze(invalidImage)).rejects.toThrow();
  });
});
```

## Documentation

### What to Document

- Complex algorithms and their logic
- API endpoints with examples
- Configuration options
- Setup and deployment procedures
- Major architectural decisions

### Format

- Use Markdown for docs
- Include code examples
- Add diagrams where helpful
- Keep language clear and concise

## Performance

### Guidelines

- Profile before optimizing
- Focus on critical paths
- Cache expensive operations
- Use lazy loading where appropriate
- Monitor bundle size

### Tools

```bash
# Check bundle size
npm run build
npm run analyze

# Profile performance
npm run profile

# Load testing
npm run load-test
```

## Security

### Best Practices

- Never commit secrets or keys
- Use environment variables
- Validate all inputs
- Use HTTPS in production
- Keep dependencies updated
- Run security audits regularly

```bash
# Check for vulnerabilities
npm audit
npm audit fix

# Update dependencies
npm outdated
npm update
```

## Issue Management

### Reporting Bugs

Include:
- Clear description of problem
- Steps to reproduce
- Expected vs actual behavior
- Environment info (OS, Node version, etc.)
- Screenshots/logs if applicable

### Feature Requests

Include:
- Clear description of desired feature
- Use case and benefits
- Proposed solution (if any)
- Any relevant context

## Questions?

- Check existing issues and documentation
- Ask in GitHub Discussions
- Open an issue for clarification
- Contact maintainers

---

**Thank you for contributing!**

Your efforts help make this project better for everyone.

Last Updated: November 2025
