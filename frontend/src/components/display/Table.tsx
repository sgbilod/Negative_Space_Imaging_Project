import React, { useState } from 'react';
import {
  Table as MuiTable,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  Checkbox,
  Paper,
  Box,
  Typography,
  useTheme,
} from '@mui/material';

export interface TableColumn<T> {
  id: keyof T;
  label: string;
  sortable?: boolean;
  format?: (value: any) => React.ReactNode;
  width?: string;
  align?: 'left' | 'center' | 'right';
}

interface TableProps<T extends Record<string, any>> {
  columns: TableColumn<T>[];
  rows: T[];
  rowKey: keyof T;
  title?: string;
  selectable?: boolean;
  onSelectRow?: (row: T) => void;
  onSelectMultiple?: (rows: T[]) => void;
  sortable?: boolean;
  paginated?: boolean;
  pageSize?: number;
  loading?: boolean;
  dense?: boolean;
}

/**
 * Table Component
 *
 * Advanced data table with:
 * - Sorting functionality
 * - Pagination
 * - Row selection
 * - Multi-select support
 * - Custom formatting
 * - Responsive layout
 */
function Table<T extends Record<string, any>>({
  columns,
  rows,
  rowKey,
  title,
  selectable = false,
  onSelectRow,
  onSelectMultiple,
  sortable = true,
  paginated = true,
  pageSize = 10,
  loading = false,
  dense = false,
}: TableProps<T>): React.ReactElement {
  const theme = useTheme();
  const [page, setPage] = useState(0);
  const [order, setOrder] = useState<'asc' | 'desc'>('asc');
  const [orderBy, setOrderBy] = useState<keyof T | ''>('');
  const [selectedRows, setSelectedRows] = useState<Set<any>>(new Set());

  const handleRequestSort = (property: keyof T) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  const handleSelectAll = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.checked) {
      const newSelected = new Set(rows.map((row) => row[rowKey]));
      setSelectedRows(newSelected);
      if (onSelectMultiple) {
        onSelectMultiple(rows.filter((row) => newSelected.has(row[rowKey])));
      }
    } else {
      setSelectedRows(new Set());
      if (onSelectMultiple) {
        onSelectMultiple([]);
      }
    }
  };

  const handleSelectRow = (row: T) => {
    const key = row[rowKey];
    const newSelected = new Set(selectedRows);

    if (newSelected.has(key)) {
      newSelected.delete(key);
    } else {
      newSelected.add(key);
    }

    setSelectedRows(newSelected);

    if (onSelectMultiple) {
      onSelectMultiple(rows.filter((r) => newSelected.has(r[rowKey])));
    }

    if (onSelectRow) {
      onSelectRow(row);
    }
  };

  const sortedRows = React.useMemo(() => {
    if (!sortable || !orderBy) return rows;

    return [...rows].sort((a, b) => {
      const aValue = a[orderBy];
      const bValue = b[orderBy];

      if (aValue < bValue) return order === 'asc' ? -1 : 1;
      if (aValue > bValue) return order === 'asc' ? 1 : -1;
      return 0;
    });
  }, [rows, orderBy, order, sortable]);

  const paginatedRows = paginated
    ? sortedRows.slice(page * pageSize, page * pageSize + pageSize)
    : sortedRows;

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  if (rows.length === 0 && !loading) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="text.secondary">No data available</Typography>
      </Paper>
    );
  }

  return (
    <Box>
      {title && (
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          {title}
        </Typography>
      )}

      <TableContainer component={Paper}>
        <MuiTable size={dense ? 'small' : 'medium'}>
          <TableHead>
            <TableRow sx={{ backgroundColor: theme.palette.action.hover }}>
              {selectable && (
                <TableCell padding="checkbox">
                  <Checkbox
                    indeterminate={
                      selectedRows.size > 0 && selectedRows.size < paginatedRows.length
                    }
                    checked={paginatedRows.length > 0 && selectedRows.size === paginatedRows.length}
                    onChange={handleSelectAll}
                  />
                </TableCell>
              )}
              {columns.map((column) => (
                <TableCell
                  key={String(column.id)}
                  align={column.align || 'left'}
                  width={column.width}
                  sx={{ fontWeight: 600 }}
                >
                  {sortable && column.sortable !== false ? (
                    <TableSortLabel
                      active={orderBy === column.id}
                      direction={orderBy === column.id ? order : 'asc'}
                      onClick={() => handleRequestSort(column.id)}
                    >
                      {column.label}
                    </TableSortLabel>
                  ) : (
                    column.label
                  )}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {paginatedRows.map((row) => {
              const isSelected = selectedRows.has(row[rowKey]);

              return (
                <TableRow
                  key={String(row[rowKey])}
                  hover
                  selected={isSelected}
                  sx={{
                    cursor: selectable ? 'pointer' : 'default',
                    backgroundColor: isSelected ? theme.palette.action.selected : 'inherit',
                  }}
                  onClick={() => selectable && handleSelectRow(row)}
                >
                  {selectable && (
                    <TableCell padding="checkbox" onClick={(e) => e.stopPropagation()}>
                      <Checkbox checked={isSelected} />
                    </TableCell>
                  )}
                  {columns.map((column) => (
                    <TableCell key={String(column.id)} align={column.align || 'left'}>
                      {column.format ? column.format(row[column.id]) : row[column.id]}
                    </TableCell>
                  ))}
                </TableRow>
              );
            })}
          </TableBody>
        </MuiTable>
      </TableContainer>

      {paginated && (
        <TablePagination
          rowsPerPageOptions={[5, 10, 25, 50]}
          component="div"
          count={sortedRows.length}
          rowsPerPage={pageSize}
          page={page}
          onPageChange={handleChangePage}
        />
      )}
    </Box>
  );
}

export default Table;
