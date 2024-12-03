<script>
  import SearchTree from './lib/search-tree.svelte'
  import { PaneGroup, Pane, PaneResizer } from 'paneforge'
  import ResizerHandle from './lib/resizer-handle.svelte'
  import PeriodicTable from './lib/periodic-table/table.svelte'
  import Info from './lib/info.svelte'
  import Structure from './lib/structure.svelte'
</script>

<div class="flex h-screen flex-col bg-gray-950 text-white">
  <header class="z-10 flex w-full flex-wrap border-b border-gray-600 shadow-xl">
    <nav class="relative w-full px-3 py-2">
      <div class="flex items-center justify-between">
        <a
          href="/"
          class="font-title flex items-center text-xl font-thin uppercase"
        >
          <span class="text-yellow-500">Chem</span>
          <span class="text-slate-400">Reasoner</span>
          <!-- &nbsp;&nbsp;
          <span class="text-sm lowercase text-red-500">
            [ for demonstration only ]
          </span> -->
        </a>
      </div>
    </nav>
  </header>
  <main class="flex-grow font-extralight">
    <PaneGroup direction="vertical">
      <Pane defaultSize={50}>
        <PaneGroup direction="horizontal">
          <Pane defaultSize={50}>
            <div class="h-full w-full bg-gray-800 p-2">
              <SearchTree />
            </div>
          </Pane>
          <PaneResizer>
            <ResizerHandle vertical={true} />
          </PaneResizer>
          <Pane defaultSize={50}>
            <div class="h-full w-full bg-gray-950 p-2">
              <Info />
            </div>
          </Pane>
        </PaneGroup>
      </Pane>
      <PaneResizer>
        <ResizerHandle vertical={false} />
      </PaneResizer>
      <Pane defaultSize={50}>
        <PaneGroup direction="horizontal">
          <Pane defaultSize={70}>
            <div class="h-full w-full bg-gray-900 p-2">
              <PeriodicTable />
            </div>
          </Pane>
          <PaneResizer>
            <ResizerHandle vertical={true} />
          </PaneResizer>
          <Pane defaultSize={30}>
            <div class="h-full w-full bg-gray-950">
              <Structure />
            </div>
          </Pane>
        </PaneGroup>
      </Pane>
    </PaneGroup>
  </main>
</div>
