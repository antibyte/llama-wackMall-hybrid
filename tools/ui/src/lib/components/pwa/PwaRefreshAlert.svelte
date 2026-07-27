<script lang="ts">
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';

	let { needRefresh: needRefreshProp, updateServiceWorker, forceReload } = $props();
	let needRefresh = $derived(needRefreshProp ?? false);
	let dismissed = $state(false);
</script>

{#if needRefresh && !dismissed}
	<Card.Root class="overflow-hidden gap-1 py-5">
		<Card.Header class="px-5">
			<Card.Title class="text-sm font-medium">Update available</Card.Title>
		</Card.Header>

		<Card.Content class="gap-6 grid px-5">
			<p class="text-xs text-muted-foreground">A new version is available. Reload to update.</p>

			<div class="flex justify-end gap-2">
				<Button size="sm" variant="outline" onclick={() => (dismissed = true)}>Not now</Button>
				<Button
					size="sm"
					onclick={() => {
						updateServiceWorker();

						if (forceReload) {
							window.location.reload();
						}

						needRefresh = false;
					}}
				>
					Reload
				</Button>
			</div>
		</Card.Content>
	</Card.Root>
{/if}
