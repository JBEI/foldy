/**
 * A minimal test file to verify Jest is working correctly
 */

// Simple test function
const add = (a: number, b: number): number => a + b;

describe('Basic Jest Tests', () => {
  it('adds two numbers correctly', () => {
    expect(add(2, 3)).toBe(5);
  });

  it('handles zero', () => {
    expect(add(0, 10)).toBe(10);
  });

  it('handles negative numbers', () => {
    expect(add(-5, 3)).toBe(-2);
  });
});
