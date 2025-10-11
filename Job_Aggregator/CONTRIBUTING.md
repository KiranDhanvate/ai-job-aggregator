# 🤝 Contributing to AI Job Aggregator & Recommender API

Thank you for your interest in contributing to this project! We welcome contributions from the community and appreciate your help in making this project better.

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Poetry (for dependency management)
- Git
- Basic understanding of FastAPI, Machine Learning, and Web Scraping

### Development Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/ai-job-aggregator.git
   cd ai-job-aggregator
   ```

2. **Install Dependencies**
   ```bash
   # Install Poetry if not already installed
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Install all dependencies including development tools
   poetry install --with dev
   
   # Activate the virtual environment
   poetry shell
   ```

3. **Set Up Pre-commit Hooks**
   ```bash
   # Install pre-commit hooks for code quality
   pre-commit install
   ```

4. **Run Tests**
   ```bash
   # Ensure all tests pass
   pytest
   ```

## 📋 Types of Contributions

### 🐛 Bug Reports
- Use the GitHub issue template for bug reports
- Include steps to reproduce the issue
- Provide system information (OS, Python version, etc.)
- Include relevant error messages and logs

### ✨ Feature Requests
- Check existing issues to avoid duplicates
- Provide a clear description of the proposed feature
- Explain the use case and potential benefits
- Consider implementation complexity and impact

### 🔧 Code Contributions
- **Bug fixes**: Fix existing issues
- **New features**: Add new functionality
- **Performance improvements**: Optimize existing code
- **Documentation**: Improve docs and examples
- **Tests**: Add or improve test coverage

## 🔄 Development Workflow

### 1. Create a Feature Branch
```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b bugfix/issue-number-description
```

### 2. Make Your Changes
- Write clean, readable code
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run the test suite
pytest

# Run specific tests
pytest tests/test_your_feature.py

# Run with coverage
pytest --cov=jobspy --cov=api_server

# Run linting
pre-commit run --all-files
```

### 4. Commit Your Changes
```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add new job filtering capability

- Add support for filtering by company size
- Include unit tests for new functionality
- Update API documentation"
```

### 5. Push and Create Pull Request
```bash
# Push your branch
git push origin feature/your-feature-name

# Create a Pull Request on GitHub
```

## 📝 Coding Standards

### Python Code Style
- Follow **PEP 8** guidelines
- Use **type hints** for function parameters and return values
- Write **docstrings** for all functions and classes
- Keep functions small and focused on a single responsibility

### Example Code Style
```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class JobFilter(BaseModel):
    """Model for job filtering parameters."""
    
    search_term: str = Field(..., description="Job search term")
    location: Optional[str] = Field(None, description="Job location")
    is_remote: bool = Field(False, description="Remote work preference")
    salary_min: Optional[int] = Field(None, description="Minimum salary")
    
    def to_scraper_params(self) -> Dict[str, Any]:
        """Convert filter to scraper parameters."""
        return {
            "search_term": self.search_term,
            "location": self.location,
            "is_remote": self.is_remote,
        }
```

### API Design Principles
- Use **RESTful** conventions
- Provide **clear error messages**
- Include **comprehensive documentation**
- Follow **consistent naming** conventions
- Use **appropriate HTTP status codes**

### Test Requirements
- Write **unit tests** for new functionality
- Include **integration tests** for API endpoints
- Aim for **>80% test coverage**
- Use **descriptive test names**

Example test:
```python
import pytest
from fastapi.testclient import TestClient
from api_server import app

client = TestClient(app)

def test_job_scraping_endpoint():
    """Test job scraping endpoint returns valid response."""
    response = client.post("/scrape", json={
        "search_term": "python developer",
        "location": "San Francisco",
        "results_wanted": 10
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert "count" in data
    assert data["count"] >= 0
```

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_scraper.py

# Run with coverage report
pytest --cov=jobspy --cov-report=html
```

### Writing Tests
- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test API endpoints and workflows
- **Mock external dependencies**: Don't make real API calls in tests
- **Test edge cases**: Include boundary conditions and error scenarios

## 📚 Documentation

### Code Documentation
- Use **Google-style docstrings** for functions and classes
- Include **type hints** for better IDE support
- Document **complex algorithms** and business logic
- Keep **README.md** updated with new features

### API Documentation
- Use **FastAPI automatic documentation**
- Provide **example requests and responses**
- Document **error codes and messages**
- Include **authentication requirements**

## 🚀 Pull Request Process

### Before Submitting
1. **Ensure all tests pass**
2. **Run linting and fix any issues**
3. **Update documentation** if needed
4. **Add tests** for new functionality
5. **Update CHANGELOG.md** with your changes

### Pull Request Template
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
```

### Review Process
1. **Automated checks** must pass (tests, linting, etc.)
2. **Code review** by maintainers
3. **Testing** in development environment
4. **Documentation review**
5. **Approval and merge**

## 🐛 Reporting Issues

### Bug Report Template
```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. Ubuntu 20.04]
 - Python version: [e.g. 3.10.0]
 - Package version: [e.g. 2.1.0]

**Additional context**
Add any other context about the problem here.
```

## 💡 Feature Request Template
```markdown
**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

## 🏷️ Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

### Examples
```bash
feat(api): add job filtering by company size
fix(scraper): handle timeout errors gracefully
docs(readme): update installation instructions
test(api): add integration tests for scraping endpoint
```

## 🎯 Areas for Contribution

### High Priority
- **Performance optimization** for large-scale scraping
- **Enhanced error handling** and retry mechanisms
- **Additional job sites** support
- **Improved ML models** for recommendations
- **API rate limiting** and caching

### Medium Priority
- **Frontend dashboard** development
- **Mobile app** integration
- **Advanced analytics** and reporting
- **Multi-language support**
- **Chrome extension** development

### Low Priority
- **Documentation improvements**
- **Test coverage** enhancements
- **Code refactoring**
- **Docker optimization**
- **CI/CD improvements**

## 🤔 Questions?

If you have any questions about contributing, feel free to:
- Open a [GitHub Discussion](https://github.com/KiranDhanvate/ai-job-aggregator/discussions)
- Contact the maintainers directly
- Check existing issues and pull requests for similar questions

## 🙏 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

Thank you for contributing to make this project better! 🚀
