# ⚛️ MASTER PROMPT 03A: REACT FRONTEND - PAGES & COMPONENTS
**Weeks 3-4 Execution (Part A) | Dec 2-6, 2025**

**Copy this entire prompt and paste into GitHub Copilot Chat, then request:**
```
"Generate files 1-20 (MainLayout.tsx through AnalysisResultsCard.tsx)"
```

---

## DETAILED PROMPT FOR COPILOT

```
TASK: Create React 18 + TypeScript frontend for Negative Space Imaging Project.

PART A: Layouts, Pages, and Components (Base Setup)

REQUIREMENTS:
- Use React 18 with TypeScript
- Use Tailwind CSS for styling
- Use React Router v6 for navigation
- Responsive mobile-first design
- Accessibility (WCAG 2.1 AA)
- Component-based architecture

FILES TO GENERATE (Part A - 1-20):

1. src/frontend/layouts/MainLayout.tsx (100 lines)
   - Main app layout component
   - Navbar, sidebar, main content area
   - User menu with logout
   - Navigation links

2. src/frontend/layouts/AuthLayout.tsx (80 lines)
   - Authentication pages layout
   - Centered form layout
   - No navigation bar

3. src/frontend/pages/LoginPage.tsx (100 lines)
   - Email and password inputs
   - Login button with loading state
   - Link to register page
   - Error message display
   - Form validation with Zod

4. src/frontend/pages/RegisterPage.tsx (120 lines)
   - Email, password, confirm password, first name, last name inputs
   - Register button with loading state
   - Link to login page
   - Password strength indicator
   - Form validation

5. src/frontend/pages/DashboardPage.tsx (100 lines)
   - User welcome message
   - Statistics cards (total images, total analyses, etc.)
   - Recent images grid
   - Quick actions (upload, view all)
   - Responsive layout

6. src/frontend/pages/ImagesPage.tsx (150 lines)
   - Gallery of user's uploaded images
   - Pagination support
   - Search/filter functionality
   - Image cards with thumbnails
   - Load more button

7. src/frontend/pages/ImageDetailPage.tsx (120 lines)
   - Full image display
   - Image metadata
   - Analysis results section
   - Delete button
   - Navigate to analysis results

8. src/frontend/pages/UploadPage.tsx (120 lines)
   - Drag-and-drop file upload area
   - File input with validation
   - Progress bar during upload
   - Preview of selected image
   - Upload button with loading state

9. src/frontend/pages/AnalysisPage.tsx (150 lines)
   - Display analysis results
   - Original image
   - Visualization with negative space highlighted
   - Statistics and metrics
   - Download results button

10. src/frontend/pages/ProfilePage.tsx (100 lines)
    - User profile information
    - Edit profile form
    - Password change form
    - Account deletion option
    - Save changes button

11. src/frontend/pages/NotFoundPage.tsx (50 lines)
    - 404 error page
    - Link back to home

12. src/frontend/components/Button.tsx (40 lines)
    - Reusable button component
    - Variants: primary, secondary, danger
    - Sizes: sm, md, lg
    - Loading state
    - Disabled state

13. src/frontend/components/Input.tsx (50 lines)
    - Reusable input component
    - Text, email, password types
    - Label and error message
    - Validation styling
    - Accessibility attributes

14. src/frontend/components/Modal.tsx (80 lines)
    - Reusable modal component
    - Title, body, footer
    - Close button
    - Overlay click to close
    - Keyboard escape to close

15. src/frontend/components/Alert.tsx (60 lines)
    - Alert message component
    - Types: success, error, warning, info
    - Dismissible
    - Icon support

16. src/frontend/components/Spinner.tsx (40 lines)
    - Loading spinner component
    - Sizes: sm, md, lg
    - Used in buttons, pages

17. src/frontend/components/Navbar.tsx (100 lines)
    - Top navigation bar
    - Logo and brand name
    - Navigation links
    - User menu dropdown
    - Mobile hamburger menu

18. src/frontend/components/ImageCard.tsx (80 lines)
    - Card displaying image thumbnail
    - Image name and upload date
    - File size
    - Delete button
    - Click to view details

19. src/frontend/components/AnalysisResultsCard.tsx (100 lines)
    - Card displaying analysis results
    - Key metrics
    - Small visualization preview
    - View full results link

20. src/frontend/components/ProtectedRoute.tsx (60 lines)
    - Route wrapper for authentication
    - Redirect to login if not authenticated
    - Require specific roles if needed

STRUCTURE EXAMPLE:
```typescript
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Button from '@/components/Button';
import Input from '@/components/Input';

export const LoginPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      // Login logic
      navigate('/dashboard');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto">
      <form onSubmit={handleSubmit}>
        <Input
          type="email"
          label="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <Input
          type="password"
          label="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <Button type="submit" disabled={loading}>
          {loading ? 'Logging in...' : 'Login'}
        </Button>
      </form>
    </div>
  );
};
```

Generate files 1-20 now.
```

---

## 📋 EXECUTION CHECKLIST

After Copilot generates the files:

- [ ] All 20 files generated successfully
- [ ] Files saved to correct directories
- [ ] No TypeScript compilation errors
- [ ] All pages render without errors
- [ ] Routing works properly
- [ ] Tailwind styling applied
- [ ] Files committed to git

## 📍 FILE STRUCTURE

```
src/frontend/
├── layouts/
│   ├── MainLayout.tsx
│   └── AuthLayout.tsx
├── pages/
│   ├── LoginPage.tsx
│   ├── RegisterPage.tsx
│   ├── DashboardPage.tsx
│   ├── ImagesPage.tsx
│   ├── ImageDetailPage.tsx
│   ├── UploadPage.tsx
│   ├── AnalysisPage.tsx
│   ├── ProfilePage.tsx
│   └── NotFoundPage.tsx
└── components/
    ├── Button.tsx
    ├── Input.tsx
    ├── Modal.tsx
    ├── Alert.tsx
    ├── Spinner.tsx
    ├── Navbar.tsx
    ├── ImageCard.tsx
    ├── AnalysisResultsCard.tsx
    └── ProtectedRoute.tsx
```

## 🔄 NEXT: MASTER PROMPT 03B

After completing this prompt:
1. Review all 20 files for correctness
2. Ensure pages render without errors
3. Commit changes to git
4. Use **MASTER_PROMPT_03B** for hooks & services

---

**Status:** Ready for execution
**Created:** November 8, 2025
**Time to Generate:** 15-20 minutes
**Part:** A of 2 for Weeks 3-4
