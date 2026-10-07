import { expect, it, vi } from "vitest";
import { createSkillWorkshopTool } from "../../agents/tools/skill-workshop-tool.js";
import { getRuntimeConfig } from "../../config/config.js";
import {
  loadExactSessionEntryReadOnly,
  upsertSessionEntryCore,
} from "../../config/sessions/session-accessor.js";
import { ensureProfileForEmail } from "../../state/user-profiles.js";
import { withOpenClawTestState } from "../../test-utils/openclaw-test-state.js";
import { createDirectChatContext } from "../server-chat.agent-events.test-helpers.js";
import { resolveSessionMutationAuthorization } from "../session-sharing.js";
import * as chatDispatch from "./chat-send-agent-dispatch.js";
import { handleDirectExternalChatSend } from "./chat-send-external-entry.js";
import type { GatewayClient } from "./types.js";

it("preserves human Workshop authoring after reconnect without granting synthetic or internal authority", async () => {
  await withOpenClawTestState({ scenario: "minimal" }, async (state) => {
    const profile = ensureProfileForEmail("reconnect-author@example.test");
    const sessionKey = "agent:main:reconnect-authoring";
    const sessionId = "preserved-session";
    const scope = { agentId: "main", sessionKey };
    await upsertSessionEntryCore(scope, {
      sessionId,
      updatedAt: Date.now(),
      status: "done",
      createdActor: { type: "human", source: "profile", id: profile.id },
    });
    const client: GatewayClient = {
      connId: "reconnect-authoring",
      connectionSignal: new AbortController().signal,
      internal: { authenticatedControlUi: true, controlUiAdmin: true },
      authenticatedUserProfile: {
        profileId: profile.id,
        displayName: null,
        hasAvatar: false,
        updatedAt: 1,
      },
      connect: {
        minProtocol: 1,
        maxProtocol: 1,
        role: "operator",
        scopes: ["operator.admin"],
        client: {
          id: "openclaw-control-ui",
          version: "test",
          platform: "web",
          mode: "webchat",
        },
      },
    };
    const context = createDirectChatContext({ getRuntimeConfig });
    const observed: Array<{
      resume: boolean;
      capability: boolean;
      personalDescription: boolean;
      sessionId: string | undefined;
      rawMessage: string;
    }> = [];
    const dispatch = vi.spyOn(chatDispatch, "startChatDispatch").mockImplementation((owned) => {
      const workshop = createSkillWorkshopTool({
        workspaceDir: state.workspaceDir,
        config: owned.session.cfg,
        agentId: "main",
        libraryAuthoring: owned.skillLibraryAuthoring,
      });
      observed.push({
        resume: owned.request.reconnectResumeRequested,
        capability: owned.skillLibraryAuthoring !== undefined,
        personalDescription: workshop.description.includes("Personal actions:"),
        sessionId: owned.session.entry?.sessionId,
        rawMessage: owned.request.rawMessage,
      });
      owned.admission.cleanupAdmittedRun();
    });
    try {
      for (const kind of ["resume", "ordinary", "synthetic", "internal", "expired"] as const) {
        const resume = kind !== "ordinary";
        const idempotencyKey = `new-${kind}-message`;
        const sendingClient: GatewayClient =
          kind === "synthetic"
            ? { ...client, internal: { ...client.internal, syntheticClient: true } }
            : kind === "internal"
              ? { ...client, internal: { ...client.internal, pluginRuntimeOwnerId: "test-plugin" } }
              : client;
        const params = {
          sessionKey,
          sessionId,
          message: `New ${kind} message.`,
          idempotencyKey,
          ...(resume ? { __controlUiReconnectResume: true } : {}),
        };
        const authorization = resolveSessionMutationAuthorization({
          client: sendingClient,
          context,
          method: "chat.send",
          requestParams: params,
        });
        expect(authorization.error).toBeNull();
        const respond = vi.fn();
        const send = handleDirectExternalChatSend({
          params,
          req: { type: "req", id: idempotencyKey, method: "chat.send", params },
          respond,
          context,
          client: sendingClient,
          hasCurrentClientAuthority: () => kind !== "expired",
          sessionMutationAuthorization: authorization.authorization,
          isWebchatConnect: () => false,
        });
        if (kind === "expired") {
          await expect(send).rejects.toThrow("Gateway caller authority is no longer active.");
          expect(respond).not.toHaveBeenCalled();
          expect(observed).toHaveLength(4);
        } else {
          await send;
          expect(respond).toHaveBeenCalledWith(
            true,
            expect.objectContaining({ status: "started", runId: idempotencyKey }),
            undefined,
            expect.anything(),
          );
        }
        expect(loadExactSessionEntryReadOnly(scope)?.entry.sessionId).toBe(sessionId);
      }
      expect(observed.map((turn) => turn.resume)).toEqual([true, false, true, true]);
      expect(observed.map((turn) => turn.rawMessage)).toEqual([
        "New resume message.",
        "New ordinary message.",
        "New synthetic message.",
        "New internal message.",
      ]);
      expect(observed.map((turn) => turn.sessionId)).toEqual([
        sessionId,
        sessionId,
        sessionId,
        sessionId,
      ]);
      expect(observed[2]).toMatchObject({ capability: false, personalDescription: false });
      expect(observed[3]).toMatchObject({ capability: false, personalDescription: false });
      expect(observed[1]).toMatchObject({ capability: true, personalDescription: true });
      expect(observed[0]).toMatchObject({ capability: true, personalDescription: true });
    } finally {
      dispatch.mockRestore();
    }
  });
});
