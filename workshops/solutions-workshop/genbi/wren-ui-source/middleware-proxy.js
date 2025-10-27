import { NextResponse } from 'next/server';

export function middleware(request) {
  const { pathname } = request.nextUrl;
  
  // Rewrite requests without basePath to include basePath
  // This allows internal container communication to work
  if ((pathname.startsWith('/api/') || pathname.startsWith('/images/')) 
      && !pathname.startsWith('/absproxy/3000/')) {
    const url = request.nextUrl.clone();
    url.pathname = `/absproxy/3000${pathname}`;
    // Use rewrite (not redirect) to keep the URL unchanged for the client
    return NextResponse.rewrite(url);
  }
  
  return NextResponse.next();
}

export const config = {
  // Match API and images paths
  matcher: ['/api/:path*', '/images/:path*'],
};
