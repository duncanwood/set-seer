'use client';

import dynamic from 'next/dynamic';

// Dynamic import to avoid SSR issues with react-webcam and onnxruntime-web
const Camera = dynamic(() => import('@/components/Camera'), { ssr: false });

export default function Home() {
  return (
    <main className="app-root">
      <Camera targetFps={15} />
    </main>
  );
}
