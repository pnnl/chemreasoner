import { writable } from 'svelte/store'

export async function fetchSearchData() {
  const r = await fetch(import.meta.env.VITE_SEARCH_TREE_URL)
  return await r.json()
}

export const nodes = writable([])
export const selectedNode = writable(null)
export const selectedElement = writable(null)

export async function fetchStructures(nodeId) {
  const r = await fetch(import.meta.env.VITE_STRUCTURES_URL.replace('{node_id}', nodeId))
  return await r.json()
}

export async function postPrompt(nodeId, prompt) {
  const r = await fetch(import.meta.env.VITE_PROMPT_URL.replace('{node_id}', nodeId), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ prompt }),
  })
  return await r.json()
}