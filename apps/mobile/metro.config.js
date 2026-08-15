const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

// Find the project and workspace directories
const projectRoot = __dirname;
const workspaceRoot = path.resolve(projectRoot, '../..');

const config = getDefaultConfig(projectRoot);

// 1. Watch all files in the monorepo workspace (including libs/shared)
config.watchFolders = [workspaceRoot];

// 2. Resolve packages from both local and root node_modules to avoid duplicate React Native instances
config.resolver.nodeModulesPaths = [
  path.resolve(projectRoot, 'node_modules'),
  path.resolve(workspaceRoot, 'node_modules'),
];

// 3. Force Metro to resolve react, react-native, and assets-registry to the mobile app's local version
config.resolver.extraNodeModules = {
  'react': path.resolve(projectRoot, 'node_modules/react'),
  'react-native': path.resolve(projectRoot, 'node_modules/react-native'),
  '@react-native/assets-registry': path.resolve(projectRoot, 'node_modules/@react-native/assets-registry'),
};

module.exports = config;
