export default function imageLoader({ src, width, quality }) {
  // Add basePath to all image sources
  const basePath = '/absproxy/3000';
  
  // If src already includes the basePath, return as-is
  if (src.startsWith(basePath)) {
    return src;
  }
  
  // Add basePath to the src
  return `${basePath}${src}`;
}
