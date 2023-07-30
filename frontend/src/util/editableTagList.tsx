import React from "react";
import { AiOutlineCloseCircle, AiOutlinePlus } from "react-icons/ai";
import UIkit from "uikit";

export interface EditableTagListProps {
  tags: string[];
  addTag: (tag: string) => void;
  deleteTag: (tag: string) => void;
  handleTagClick: (tag: string) => void;
}

export function EditableTagList(props: EditableTagListProps) {
  const addNewTag = () => {
    UIkit.modal.prompt("New tag:", "").then(
      (newTag: string | null) => {
        if (newTag) {
          const allowedCharsRegex = /^[a-zA-Z0-9_-]+$/;
          if (allowedCharsRegex.test(newTag)) {
            props.addTag(newTag);
          } else {
            UIkit.notification(
              `Invalid tag: ${newTag} contains a character which is not a letter, number, hyphen, or underscore.`
            );
          }
        }
      },
      () => {
        console.log("No new tag.");
      }
    );
  };

  return (
    <div
      className="uk-input uk-text-nowrap hiddenscrollbar" // uk-text-truncate
      onInput={(e) => console.log(e)}
      style={{ borderRadius: "100px", overflow: "scroll" }}
    >
      <span className="uk-margin-small-right uk-text-light">Tags:</span>
      {props.tags.map((tag: string) => {
        return (
          <span
            key={tag}
            className="uk-badge uk-badge-bigger uk-margin-small-right"
            uk-tooltip="Tags help sort and manage collections or batches of folds. Tags must only contain letters."
          >
            <span
              style={{ padding: "0 3px 0 8px" }}
              onClick={() => props.handleTagClick(tag)}
            >
              {tag}
            </span>
            <AiOutlineCloseCircle
              style={{ cursor: "pointer" }}
              onClick={() => props.deleteTag(tag)}
            />
          </span>
        );
      })}
      <span
        className="uk-badge uk-badge-bigger uk-margin-small-right"
        style={{ background: "#999999", cursor: "pointer" }}
        onClick={() => addNewTag()}
      >
        <AiOutlinePlus />
      </span>
    </div>
  );
}
