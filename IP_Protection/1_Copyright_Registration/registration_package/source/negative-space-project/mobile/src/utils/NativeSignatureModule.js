// NativeSignatureModule.js
// Placeholder for native module integration (Android/iOS)

import { NativeModules } from 'react-native';

const { SignatureNative } = NativeModules;

export default {
  extractSignature: async (imageData) => {
    if (SignatureNative && SignatureNative.extractSignature) {
      return await SignatureNative.extractSignature(imageData);
    }
    throw new Error('Native signature extraction not available');
  },
  verifySignature: async (signatureData) => {
    if (SignatureNative && SignatureNative.verifySignature) {
      return await SignatureNative.verifySignature(signatureData);
    }
    throw new Error('Native signature verification not available');
  }
};
