/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',

  // Configurable base path for GitHub Pages deployment
  // If running locally, leave empty. If deploying to user.github.io/repo, set /repo
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || '',

  // Empty turbopack config to satisfy Next.js 16
  turbopack: {},

  // Ensure ONNX model isn't processed by the bundler
  serverExternalPackages: ['onnxruntime-web'],
};

module.exports = nextConfig;
