/**
 * Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
 * PROPRIETARY AND CONFIDENTIAL - PATENT PENDING
 * 
 * Quantum Test Utilities
 * Support functions for quantum testing framework
 */

const crypto = require('crypto');
const { performance } = require('perf_hooks');

class QuantumTestUtils {
    static generateQuantumHash(data) {
        return crypto.createHash('sha512')
            .update(typeof data === 'string' ? data : JSON.stringify(data))
            .digest('hex');
    }

    static measureQuantumTime() {
        return BigInt(Math.floor(performance.now() * 1000000));
    }

    static createQuantumSignature() {
        return {
            timestamp: this.measureQuantumTime(),
            entropy: crypto.randomBytes(32),
            coherence: Math.random() * 0.5 + 0.5
        };
    }

    static validateQuantumSignature(signature) {
        const currentTime = this.measureQuantumTime();
        const timeDrift = Number(currentTime - signature.timestamp) / 1000000;
        
        return {
            valid: timeDrift < 1000 && signature.coherence >= 0.5,
            timeDrift,
            coherence: signature.coherence
        };
    }

    static async sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    static async withTimeout(promise, timeoutMs = 5000) {
        const timeout = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Quantum operation timeout')), timeoutMs);
        });
        
        return Promise.race([promise, timeout]);
    }

    static measureCoherence(values) {
        if (!Array.isArray(values) || values.length === 0) {
            return 0;
        }
        
        const sum = values.reduce((acc, val) => acc + (typeof val === 'number' ? val : 0), 0);
        return sum / values.length;
    }

    static generateTestData(size = 1000) {
        const data = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            data[i] = Math.random();
        }
        return data;
    }
}

module.exports = QuantumTestUtils;
