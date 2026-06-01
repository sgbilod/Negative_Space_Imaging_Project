/**
 * QUICK REFERENCE GUIDE - Phase 6 & 7 Components
 * ================================================
 *
 * Use this guide to quickly reference how to use each component
 * in your application.
 */

// ============================================================================
// LAYOUT COMPONENTS - frontend/src/components/layout/index.ts
// ============================================================================

/**
 * MainLayout - Root layout wrapper for your entire app
 *
 * Features: Responsive sidebar, fixed navbar, sticky footer
 *
 * Usage:
 * ------
 * function App() {
 *   return (
 *     <MainLayout>
 *       <Dashboard />
 *       <Analytics />
 *     </MainLayout>
 *   );
 * }
 */

/**
 * NavigationBar - Top navigation bar with user profile menu
 *
 * Features: Menu toggle, app title, user avatar, logout
 * Automatically integrated into MainLayout
 *
 * Props:
 * - onMenuToggle?: () => void - Callback when menu button clicked
 * - elevation?: number - Shadow depth
 */

/**
 * Sidebar - Navigation menu with role-based access control
 *
 * Features: Active route highlighting, RBAC, collapsible
 * Automatically integrated into MainLayout
 *
 * Props:
 * - open: boolean - Whether sidebar is visible
 * - onClose: () => void - Close handler
 * - width: number - Sidebar width in pixels
 * - mobile?: boolean - Mobile drawer mode
 */

/**
 * Footer - Application footer with links and copyright
 *
 * Features: Company info, social links, responsive layout
 * Automatically integrated into MainLayout
 *
 * Props:
 * - companyName?: string - Default: 'Negative Space Imaging'
 * - year?: number - Copyright year, default: current year
 */

/**
 * PrivateRoute - Protected route wrapper for authenticated users
 *
 * Features: Auth check, loading states, role-based access
 *
 * Usage:
 * ------
 * <Routes>
 *   <Route element={<PrivateRoute>Dashboard</PrivateRoute>} path="/dashboard" />
 *   <Route element={<PrivateRoute requiredRoles={['admin']}>Users</PrivateRoute>} path="/users" />
 * </Routes>
 */

// ============================================================================
// FORM COMPONENTS - frontend/src/components/form/index.ts
// ============================================================================

/**
 * TextField - MUI TextField wrapper with validation
 *
 * Features: React Hook Form integration, character count, adornments
 *
 * Usage:
 * ------
 * const { register, formState: { errors } } = useForm();
 *
 * <TextField
 *   label="Username"
 *   {...register('username')}
 *   error={errors.username}
 *   characterLimit={50}
 *   showCharCount
 * />
 *
 * Props:
 * - label?: string
 * - error?: FieldError
 * - characterLimit?: number
 * - showCharCount?: boolean
 * - startAdornment?: ReactNode
 * - endAdornment?: ReactNode
 */

/**
 * Select - Dropdown select with grouped options
 *
 * Features: Flat or grouped options, loading state, React Hook Form integration
 *
 * Usage:
 * ------
 * <Select
 *   label="Category"
 *   options={[
 *     { label: 'Option 1', value: '1' },
 *     { label: 'Option 2', value: '2' },
 *   ]}
 *   value={category}
 *   onChange={(e) => setCategory(e.target.value)}
 * />
 *
 * // With groups:
 * <Select
 *   label="Category"
 *   groupedOptions={{
 *     'Fruits': [
 *       { label: 'Apple', value: 'apple' },
 *       { label: 'Banana', value: 'banana' },
 *     ],
 *     'Vegetables': [
 *       { label: 'Carrot', value: 'carrot' },
 *     ],
 *   }}
 * />
 */

/**
 * Checkbox - Single checkbox with label and validation
 *
 * Usage:
 * ------
 * <Checkbox
 *   label="I agree to terms"
 *   {...register('agreed')}
 *   error={errors.agreed}
 * />
 */

/**
 * CheckboxGroup - Multiple checkboxes with grouping
 *
 * Usage:
 * ------
 * <CheckboxGroup
 *   legend="Select permissions"
 *   options={[
 *     { label: 'Read', value: 'read' },
 *     { label: 'Write', value: 'write' },
 *     { label: 'Delete', value: 'delete' },
 *   ]}
 *   selected={permissions}
 *   onChange={setPermissions}
 * />
 */

/**
 * Radio - Single radio button with label
 *
 * Usage:
 * ------
 * <Radio
 *   label="Option 1"
 *   {...register('choice')}
 *   value="option1"
 * />
 */

/**
 * RadioGroup - Multiple radio buttons with descriptions
 *
 * Usage:
 * ------
 * <RadioGroup
 *   legend="Choose plan"
 *   options={[
 *     { label: 'Basic', value: 'basic', description: '$9/month' },
 *     { label: 'Pro', value: 'pro', description: '$29/month' },
 *     { label: 'Enterprise', value: 'enterprise', description: 'Custom' },
 *   ]}
 *   value={plan}
 *   onChange={setPlan}
 *   row // Display horizontally
 * />
 */

/**
 * DatePicker - Single date input with constraints
 *
 * Features: Min/max dates, weekend disabling, custom disabled dates
 *
 * Usage:
 * ------
 * <DatePicker
 *   label="Event date"
 *   minDate="2025-01-01"
 *   maxDate="2025-12-31"
 *   disableWeekends
 *   disabledDates={['2025-01-15', '2025-02-20']}
 *   {...register('eventDate')}
 * />
 */

/**
 * DateRangePicker - Start and end date inputs
 *
 * Features: Ensures end date > start date, responsive layout
 *
 * Usage:
 * ------
 * <DateRangePicker
 *   label="Project timeline"
 *   startDate={startDate}
 *   endDate={endDate}
 *   onStartDateChange={setStartDate}
 *   onEndDateChange={setEndDate}
 * />
 */

// ============================================================================
// DISPLAY COMPONENTS - frontend/src/components/display/index.ts
// ============================================================================

/**
 * Table - Advanced data table with sorting, pagination, selection
 *
 * Features: Sortable columns, pagination, multi-select rows
 *
 * Usage:
 * ------
 * const columns = [
 *   { id: 'name', label: 'Name', sortable: true },
 *   { id: 'email', label: 'Email', sortable: true },
 *   { id: 'role', label: 'Role', format: (val) => <Badge>{val}</Badge> },
 * ];
 *
 * <Table
 *   columns={columns}
 *   rows={users}
 *   rowKey="id"
 *   title="Users"
 *   selectable
 *   onSelectMultiple={(selected) => console.log(selected)}
 *   sortable
 *   paginated
 *   pageSize={10}
 * />
 */

/**
 * Card - Enhanced card component with sections
 *
 * Features: Image, header, content, actions, footer sections
 *
 * Usage:
 * ------
 * <Card
 *   title="User Profile"
 *   subtitle="Stephen Bilodeau"
 *   image="https://..."
 *   imageHeight={200}
 *   actions={<Button>Edit</Button>}
 *   footer="Last updated: 2025-01-15"
 *   interactive
 *   onClick={() => navigate('/user/123')}
 * >
 *   <Typography>User details go here</Typography>
 * </Card>
 */

/**
 * Gallery - Responsive image gallery with lightbox
 *
 * Features: Responsive grid, lightbox modal, keyboard navigation
 *
 * Usage:
 * ------
 * <Gallery
 *   images={[
 *     { id: 1, src: 'img1.jpg', alt: 'Image 1', title: 'First' },
 *     { id: 2, src: 'img2.jpg', alt: 'Image 2', title: 'Second' },
 *   ]}
 *   columns={4}
 *   columnsMobile={1}
 *   columnsMedium={2}
 * />
 *
 * Props:
 * - images: GalleryImage[] - Array of images
 * - columns?: number - Desktop columns (default: 4)
 * - columnsMobile?: number - Mobile columns (default: 1)
 * - columnsMedium?: number - Tablet columns (default: 2)
 */

/**
 * Badge - Status badge with semantic colors
 *
 * Features: Variants (success, error, warning, info), icons, count, pulse
 *
 * Usage:
 * ------
 * // Simple badge
 * <Badge variant="success">Active</Badge>
 *
 * // With icon
 * <Badge variant="error" icon={<ErrorIcon />}>Error</Badge>
 *
 * // With count
 * <Badge variant="info" count={42} max={99}>Notifications</Badge>
 *
 * // With pulse animation
 * <Badge variant="warning" pulse>Pending</Badge>
 */

/**
 * StatusBadge - Pre-configured status badges
 *
 * Usage:
 * ------
 * <StatusBadge status="active" />
 * <StatusBadge status="pending" />
 * <StatusBadge status="error" label="Custom label" />
 *
 * Status values: 'active' | 'inactive' | 'pending' | 'error' | 'warning'
 */

/**
 * CountBadge - Specialized count badge
 *
 * Usage:
 * ------
 * <CountBadge count={7} /> // Shows "7"
 * <CountBadge count={105} max={99} /> // Shows "99+"
 */

// ============================================================================
// COMMON PATTERNS
// ============================================================================

/**
 * Pattern 1: Complete Form with Validation
 * ────────────────────────────────────────
 */
// function MyForm() {
//   const { register, handleSubmit, formState: { errors }, watch } = useForm();
//
//   return (
//     <form onSubmit={handleSubmit(onSubmit)}>
//       <TextField
//         label="Username"
//         {...register('username', { required: 'Username required' })}
//         error={errors.username}
//       />
//       <Select
//         label="Role"
//         options={roles}
//         {...register('role')}
//         error={errors.role}
//       />
//       <CheckboxGroup
//         legend="Permissions"
//         options={permissions}
//         selected={watch('permissions')}
//         {...register('permissions')}
//       />
//       <Button type="submit">Submit</Button>
//     </form>
//   );
// }

/**
 * Pattern 2: Data Management with Table
 * ──────────────────────────────────────
 */
// function UserManagement() {
//   const [selected, setSelected] = useState([]);
//
//   const handleDelete = () => {
//     api.deleteUsers(selected.map(u => u.id));
//     setSelected([]);
//   };
//
//   return (
//     <Box>
//       <Table
//         columns={userColumns}
//         rows={users}
//         rowKey="id"
//         selectable
//         onSelectMultiple={setSelected}
//       />
//       {selected.length > 0 && (
//         <Button onClick={handleDelete}>Delete {selected.length}</Button>
//       )}
//     </Box>
//   );
// }

/**
 * Pattern 3: Responsive Gallery with Cards
 * ──────────────────────────────────────────
 */
// function ProjectShowcase() {
//   return (
//     <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 2 }}>
//       {projects.map(project => (
//         <Card
//           key={project.id}
//           title={project.name}
//           image={project.thumbnail}
//           interactive
//           onClick={() => navigate(`/project/${project.id}`)}
//         >
//           <Typography>{project.description}</Typography>
//         </Card>
//       ))}
//     </Box>
//   );
// }

// ============================================================================
// TIPS & BEST PRACTICES
// ============================================================================

/**
 * 1. Always use barrel exports for clean imports:
 *    ✓ import { TextField, Select } from '@/components/form'
 *    ✗ import TextField from '@/components/form/TextField'
 *
 * 2. Leverage React Hook Form with form components:
 *    The components are designed to work seamlessly with react-hook-form
 *
 * 3. Use TypeScript for type safety:
 *    All components are fully typed with proper interfaces
 *
 * 4. Leverage responsive design:
 *    Components auto-adjust to screen size (mobile-first approach)
 *
 * 5. Combine components for complex UIs:
 *    Example: MainLayout + Table + Dialogs for a complete dashboard
 *
 * 6. Use Material-UI theme for customization:
 *    All components respect the MUI theme configuration
 *
 * 7. Check component JSDoc in source files:
 *    Each component file has detailed documentation
 */
