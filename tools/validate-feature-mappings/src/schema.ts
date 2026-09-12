import { z } from "zod";

// Mirrors the exact shape train_risk_model.py's engineer_features() writes
// to data/models/feature_mappings.json: a global cancellation rate plus
// per-presenter and per-artist smoothed cancellation-rate lookups, keyed
// by their real (stringified) database IDs.
export const RateMapSchema = z.record(z.string(), z.number().min(0).max(1));

export const FeatureMappingsSchema = z.object({
  global_cancellation_rate: z.number().min(0).max(1),
  presenters: RateMapSchema,
  artists: RateMapSchema,
});

export type FeatureMappings = z.infer<typeof FeatureMappingsSchema>;
