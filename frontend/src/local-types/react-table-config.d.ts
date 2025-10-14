import 'react-table';
import {
  UseSortByColumnOptions,
  UseSortByColumnProps,
  UseSortByHooks,
  UseSortByInstanceProps,
  UseSortByOptions,
  UseSortByState,
} from 'react-table';

declare module 'react-table' {
  // Extend table options
  export interface TableOptions<D extends object>
    extends UseSortByOptions<D> {}

  // Extend hooks
  export interface Hooks<D extends object = {}> extends UseSortByHooks<D> {}

  // Extend instance properties
  export interface TableInstance<D extends object = {}>
    extends UseSortByInstanceProps<D> {}

  // Extend table state
  export interface TableState<D extends object = {}>
    extends UseSortByState<D> {}

  // Extend column options
  export interface ColumnInterface<D extends object = {}>
    extends UseSortByColumnOptions<D> {}

  // Extend column instance
  export interface ColumnInstance<D extends object = {}>
    extends UseSortByColumnProps<D> {}
}
