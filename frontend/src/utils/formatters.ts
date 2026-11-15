/** Small presentation helpers shared by chat and source components. */

import type { IndexedDocument } from '../types'

/** Human-readable page location; never invents a page for TXT/DOCX files. */
export function formatPageLabel(pageNumber: number | null): string | null {
  return pageNumber === null ? null : `Page ${pageNumber}`
}

/** Chunk indexes are zero-based in metadata, displayed one-based for humans. */
export function formatChunkLabel(chunkIndex: number): string {
  return `Chunk ${chunkIndex + 1}`
}

export function formatSourceLocation(
  pageNumber: number | null,
  chunkIndex: number,
): string {
  return [formatPageLabel(pageNumber), formatChunkLabel(chunkIndex)]
    .filter(Boolean)
    .join(' · ')
}

export function formatDocumentSubtitle(document: IndexedDocument): string {
  const type = document.file_type.toUpperCase()
  const chunks = `${document.chunk_count} ${document.chunk_count === 1 ? 'chunk' : 'chunks'}`
  return `${type} · ${chunks}`
}

export function truncateFileName(fileName: string, maxLength = 28): string {
  if (fileName.length <= maxLength) {
    return fileName
  }
  const extensionIndex = fileName.lastIndexOf('.')
  if (extensionIndex === -1) {
    return `${fileName.slice(0, maxLength - 1)}…`
  }
  const extension = fileName.slice(extensionIndex)
  const stemLength = Math.max(4, maxLength - extension.length - 1)
  return `${fileName.slice(0, stemLength)}…${extension}`
}
