import {
  createRootRoute,
  createRoute,
  createRouter,
  RouterProvider,
} from '@tanstack/react-router';
import { z } from 'zod';
import { App } from './App';

/** Coerce loose query values into a boolean ('1'/'true' → true, '0'/'false' →
 * false, absent/other → `dflt`). */
const flag = (dflt: boolean) =>
  z.preprocess(
    (v) =>
      v === true || v === 'true' || v === '1' || v === 1
        ? true
        : v === false || v === 'false' || v === '0' || v === 0
          ? false
          : dflt,
    z.boolean(),
  );

export const searchSchema = z.object({
  // Default to the in-browser ACT model — the app runs as a static site with no
  // backend (policy + sim are entirely client-side).
  mode: z.enum(['teleop', 'expert', 'act', 'molmo']).catch('act'),
  run: flag(false),
  cams: flag(true), // live camera panes; recording/eval pass cams=0 for speed
  cams3: flag(false), // record/stream the 3-cam set (wrist+front+side) for ACT
  molmo: z.string().optional(), // Molmo inference endpoint override
});

export type LabSearch = z.infer<typeof searchSchema>;
export type ControlMode = LabSearch['mode'];

const rootRoute = createRootRoute();

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  validateSearch: searchSchema,
  component: App,
});

const routeTree = rootRoute.addChildren([indexRoute]);
export const router = createRouter({ routeTree });

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}

export function Root() {
  return <RouterProvider router={router} />;
}
