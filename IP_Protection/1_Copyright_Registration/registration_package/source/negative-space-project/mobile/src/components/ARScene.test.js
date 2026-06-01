import React from 'react';
import { render } from '@testing-library/react-native';
import ARScene from './ARScene';

describe('ARScene', () => {
  it('renders without crashing and shows analytics overlay', () => {
    const signatureData = { features: [[0,0,0],[1,1,1],[2,2,2]] };
    const { getByText } = render(<ARScene signatureData={signatureData} />);
    expect(getByText('Negative Space Signature')).toBeTruthy();
    expect(getByText(/Points:/)).toBeTruthy();
  });
});
