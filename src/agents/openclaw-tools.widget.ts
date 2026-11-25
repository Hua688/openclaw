import { resolveCurrentChannelWidgetPresenter } from "../canvas/widget-tool.js";
import { resolveWidgetPresenters } from "../plugins/widget-presenters.js";
import { normalizeDeliveryContext } from "../utils/delivery-context.shared.js";
import type { OpenClawToolsOptions } from "./openclaw-tools.types.js";

export function resolveOpenClawWidgetContext(options: OpenClawToolsOptions | undefined) {
  const widgetDeliveryContext = normalizeDeliveryContext({
    channel: options?.agentChannel,
    to: options?.agentTo ?? options?.currentMessagingTarget ?? options?.currentChannelId,
    accountId: options?.agentAccountId,
    threadId: options?.agentThreadId,
  });
  const presenterContext = {
    messageChannel: options?.agentChannel,
    accountId: options?.agentAccountId,
    deliveryContext: widgetDeliveryContext,
    nativeChannelId: options?.nativeChannelId,
    currentChannelId: options?.currentChannelId,
    currentMessagingTarget: options?.currentMessagingTarget,
    ...(options?.currentMessageId !== undefined
      ? { currentMessageId: String(options.currentMessageId) }
      : {}),
    sessionKey: options?.runSessionKey ?? options?.agentSessionKey,
  };
  const widgetPresenters = resolveWidgetPresenters().map((registration) => registration.presenter);
  const currentChannelWidgetPresenter = resolveCurrentChannelWidgetPresenter(
    widgetPresenters,
    presenterContext,
  );
  // Scheduled runs must not borrow a remembered client as an inline renderer.
  const scheduledCaller = Boolean(options?.gatewayCallerScheduled || options?.scheduledToolTargets);
  const widgetClientCaps = scheduledCaller ? options?.clientCaps : options?.listingClientCaps;
  const inlineWidgetClientAvailable =
    (widgetClientCaps ?? options?.clientCaps)?.includes("inline-widgets") === true;
  return {
    widgetDeliveryContext,
    presenterContext,
    widgetPresenters,
    currentChannelWidgetPresenter,
    scheduledCaller,
    inlineWidgetClientAvailable,
  };
}
