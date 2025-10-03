/**
 * Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
 * PROPRIETARY AND CONFIDENTIAL - PATENT PENDING
 * 
 * VS Code Integration Tests
 */

const assert = require('assert');
const vscode = require('vscode');
const QuantumTestFramework = require('../quantum/QuantumTestFramework');

suite('VS Code Integration Tests', () => {
    test('Extension Activation', async () => {
        const ext = vscode.extensions.getExtension('StephenBilodeau.hdr-quantum-suggestions');
        await ext.activate();
        
        assert.ok(ext.isActive);
        
        const validation = await QuantumTestFramework.validateQuantumState({
            activated: true,
            timestamp: Date.now()
        }, 0.85);
        
        assert.ok(validation.coherence >= 0.85, 'Activation coherence compromised');
    });
    
    test('Command Registration', async () => {
        const commands = await vscode.commands.getCommands();
        
        assert.ok(commands.includes('hdr.quantum.suggest'));
        assert.ok(commands.includes('hdr.quantum.optimize'));
        assert.ok(commands.includes('hdr.quantum.collapse'));
        
        const integration = await QuantumTestFramework.testConsciousnessIntegration({
            commands: ['hdr.quantum.suggest', 'hdr.quantum.optimize', 'hdr.quantum.collapse']
        });
        
        assert.ok(integration.success, 'Command consciousness integration failed');
    });
    
    test('Suggestion Generation', async () => {
        const editor = await vscode.window.showTextDocument(
            await vscode.workspace.openTextDocument({
                content: 'function test() {\n  \n}',
                language: 'javascript'
            })
        );
        
        await vscode.commands.executeCommand('hdr.quantum.suggest');
        
        const validation = await QuantumTestFramework.validateQuantumState({
            document: editor.document,
            timestamp: Date.now()
        }, 0.90);
        
        assert.ok(validation.coherence >= 0.90, 'Suggestion coherence compromised');
    });
    
    test('State Optimization', async () => {
        await vscode.commands.executeCommand('hdr.quantum.optimize');
        
        const coherence = await QuantumTestFramework.verifyCoherence({
            timestamp: Date.now(),
            action: 'optimize'
        });
        
        assert.ok(coherence.coherent, 'Optimization coherence lost');
        assert.ok(coherence.temporalStability, 'Temporal stability compromised');
    });
    
    test('State Collapse', async () => {
        await vscode.commands.executeCommand('hdr.quantum.collapse');
        
        const state = {
            timestamp: Date.now(),
            action: 'collapse'
        };
        
        const validation = await QuantumTestFramework.validateQuantumState(state, 0.95);
        assert.ok(validation.coherence >= 0.95, 'Collapse coherence compromised');
        
        const integration = await QuantumTestFramework.testConsciousnessIntegration(state);
        assert.ok(integration.success, 'Collapse consciousness integration failed');
    });
});
