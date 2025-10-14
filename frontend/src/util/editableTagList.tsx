import React, { useState } from "react";
import { Tag, Input, Button, Space } from "antd";
import { PlusOutlined } from "@ant-design/icons";
import { notify } from "../services/NotificationService";

export interface EditableTagListProps {
    tags: string[];
    addTag: (tag: string) => void;
    deleteTag: (tag: string) => void;
    handleTagClick: (tag: string) => void;
    viewOnly?: boolean;
}

export function EditableTagList(props: EditableTagListProps) {
    const [inputVisible, setInputVisible] = useState(false);
    const [inputValue, setInputValue] = useState('');

    const handleInputConfirm = () => {
        if (inputValue) {
            const allowedCharsRegex = /^[a-zA-Z0-9_-]+$/;
            if (allowedCharsRegex.test(inputValue)) {
                props.addTag(inputValue);
                setInputValue('');
                setInputVisible(false);
            } else {
                notify.error(`Invalid tag: ${inputValue} contains a character which is not a letter, number, hyphen, or underscore.`);
            }
        } else {
            setInputVisible(false);
        }
    };

    const showInput = () => {
        setInputVisible(true);
    };

    return (
        <Space size={[0, 8]} >
            {props.tags.map((tag: string) => (
                <Tag
                    key={tag}
                    closable={!props.viewOnly}
                    onClose={props.viewOnly ? undefined : () => props.deleteTag(tag)}
                    style={{ paddingRight: "15px" }}
                >
                    <a
                        href={`/tag/${tag}`}
                        onClick={(e) => {
                            if (!e.ctrlKey && !e.metaKey) {
                                e.preventDefault();
                                props.handleTagClick(tag);
                            }
                        }}
                        style={{ cursor: 'pointer' }}
                    >
                        {tag}
                    </a>
                </Tag>
            ))}
            {!props.viewOnly && (
                inputVisible ? (
                    <Input
                        type="text"
                        size="small"
                        style={{ width: 100 }}
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onBlur={handleInputConfirm}
                        onPressEnter={handleInputConfirm}
                        autoFocus
                    />
                ) : (
                    <Tag onClick={showInput} style={{ background: '#fff', borderStyle: 'dashed' }}>
                        <PlusOutlined /> New Tag
                    </Tag>
                )
            )}
        </Space>
    );
}
