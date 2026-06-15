import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'LocalRAG - Private Document Q&A',
  description: '100% local AI document question answering',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full bg-gray-950 text-gray-100 antialiased font-sans">
        {children}
      </body>
    </html>
  );
}
