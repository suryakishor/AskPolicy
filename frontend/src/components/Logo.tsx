import { BookOpen } from 'lucide-react'

interface LogoProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const SIZES = {
  sm: { box: 'size-7 rounded-lg', icon: 15 },
  md: { box: 'size-9 rounded-[10px]', icon: 18 },
  lg: { box: 'size-11 rounded-xl', icon: 22 },
} as const

/** Consistent AskPolicy brand mark used in the sidebar, header, and messages. */
export function Logo({ size = 'md', className = '' }: LogoProps) {
  const config = SIZES[size]
  return (
    <span
      aria-hidden="true"
      className={`inline-flex shrink-0 items-center justify-center bg-accent text-white shadow-[0_1px_2px_rgba(28,91,79,0.25)] ${config.box} ${className}`}
    >
      <BookOpen size={config.icon} strokeWidth={1.9} />
    </span>
  )
}
