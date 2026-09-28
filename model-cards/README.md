# Lyra model card drafts

Copy each model's `README.md` to the corresponding Hugging Face model repository when that model has trained and the released artifacts have been tested:

- [Lyra-1.0-Flash](Lyra-1.0-Flash/README.md): 80,369,028 parameters, V2-S mapper.
- [Lyra-1.0](Lyra-1.0/README.md): 117,348,612 parameters, V2-L mapper candidate.

Both cards use the verified prepared-corpus report, not a claimed full-dataset training run. Before publishing weights, update the card with the actual checkpoint's dataset identity, completed training cycles, feature/encoder identity, held-out timing and mapping results, and human playtest outcomes. The current local generation path also needs matching manifests and style data; a weights-only upload is not a working standalone model. Review rights to any audio or beatmaps before uploading data or examples. No license is asserted by these drafts.
