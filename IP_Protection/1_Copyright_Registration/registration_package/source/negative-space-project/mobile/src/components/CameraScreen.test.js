import React from 'react';
import { render } from '@testing-library/react-native';
import CameraScreen from './CameraScreen';

describe('CameraScreen', () => {
  it('renders and toggles overlay', () => {
    const { getByText } = render(<CameraScreen onCapture={() => {}} />);
    expect(getByText('Show Overlay')).toBeTruthy();
  });
});
