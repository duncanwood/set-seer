/**
 * Get the full path for a static asset, accounting for the base path
 * (e.g. when deployed to GitHub Pages at /repo-name/).
 */
export function getAssetPath(path: string): string {
  const base = process.env.NEXT_PUBLIC_BASE_PATH || '';
  // If path starts with /, remove it to avoid double slash
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${base}${normalizedPath}`;
}
