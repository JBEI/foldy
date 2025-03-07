// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
// import "@testing-library/jest-dom/extend-expect";
import '@testing-library/jest-dom';  // Updated import path

// Mock matchMedia
window.matchMedia = window.matchMedia || function() {
    return {
        matches: false,
        addListener: function() {},
        removeListener: function() {},
        addEventListener: function() {},
        removeEventListener: function() {},
        dispatchEvent: function() {},
    };
};

// Mock ResizeObserver
class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
}

window.ResizeObserver = window.ResizeObserver || ResizeObserverMock;

// Mock getComputedStyle
window.getComputedStyle = window.getComputedStyle || function() {
    return {
        getPropertyValue: function() {}
    };
};


// Suppress specific Ant Design warnings
const originalError = console.error;
console.error = (...args) => {
    if (
        /Warning.*not wrapped in act/.test(args[0]) ||
        /Warning.*Received .* for a non-boolean attribute/.test(args[0])
    ) {
        return;
    }
    originalError.call(console, ...args);
};