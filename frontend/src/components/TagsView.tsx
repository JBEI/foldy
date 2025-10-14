import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
    Card,
    Row,
    Col,
    Statistic,
    Tag,
    Typography,
    Spin,
    Input,
    Space,
    Empty,
    Avatar,
    Tooltip
} from "antd";
import {
    TagOutlined,
    FolderOutlined,
    UserOutlined,
    SearchOutlined
} from "@ant-design/icons";
import { getAllTags, TagInfo } from "../api/foldApi";
import { notify } from "../services/NotificationService";

const { Title, Text } = Typography;
const { Search } = Input;

function TagsView() {
    const [tags, setTags] = useState<TagInfo[]>([]);
    const [filteredTags, setFilteredTags] = useState<TagInfo[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState("");
    const navigate = useNavigate();

    useEffect(() => {
        fetchTags();
    }, []);

    useEffect(() => {
        // Filter tags based on search term
        if (searchTerm) {
            const filtered = tags.filter(tagInfo =>
                tagInfo.tag.toLowerCase().includes(searchTerm.toLowerCase()) ||
                tagInfo.contributors.some(contributor =>
                    contributor.toLowerCase().includes(searchTerm.toLowerCase())
                )
            );
            setFilteredTags(filtered);
        } else {
            setFilteredTags(tags);
        }
    }, [tags, searchTerm]);

    const fetchTags = async () => {
        try {
            setLoading(true);
            const tagsData = await getAllTags();
            setTags(tagsData);
            setFilteredTags(tagsData);
        } catch (error) {
            console.error("Failed to fetch tags:", error);
            notify.error("Failed to load tags");
        } finally {
            setLoading(false);
        }
    };

    const handleTagClick = (tagName: string) => {
        navigate(`/tag/${tagName}`);
    };

    const getTagColor = (foldCount: number) => {
        if (foldCount >= 10) return "#f50";
        if (foldCount >= 5) return "#2db7f5";
        if (foldCount >= 2) return "#87d068";
        return "#108ee9";
    };

    const totalTags = tags.length;
    const totalFolds = tags.reduce((sum, tag) => sum + tag.fold_count, 0);
    const uniqueContributors = new Set(tags.flatMap(tag => tag.contributors)).size;

    if (loading) {
        return (
            <div style={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                minHeight: "400px"
            }}>
                <Spin size="large" tip="Loading tags..." />
            </div>
        );
    }

    return (
        <div style={{
            padding: "24px",
            width: "100%",
            maxWidth: "100%",
            overflowX: 'hidden',
            overflowY: 'auto'
        }}>
            <div style={{ textAlign: "center", marginBottom: "32px" }}>
                <Title level={2}>
                    <TagOutlined /> Browse Tags
                </Title>
                <Text type="secondary">
                    Explore all tags used in the system and discover related folds
                </Text>
            </div>

            {/* Summary Statistics */}
            <Row gutter={[16, 16]} style={{ marginBottom: "32px", maxWidth: "800px", margin: "0 auto 32px auto" }}>
                <Col xs={24} sm={8}>
                    <Card>
                        <Statistic
                            title="Total Tags"
                            value={totalTags}
                            prefix={<TagOutlined />}
                        />
                    </Card>
                </Col>
                <Col xs={24} sm={8}>
                    <Card>
                        <Statistic
                            title="Tagged Folds"
                            value={totalFolds}
                            prefix={<FolderOutlined />}
                        />
                    </Card>
                </Col>
                <Col xs={24} sm={8}>
                    <Card>
                        <Statistic
                            title="Contributors"
                            value={uniqueContributors}
                            prefix={<UserOutlined />}
                        />
                    </Card>
                </Col>
            </Row>

            {/* Search */}
            <div style={{ display: "flex", justifyContent: "center", marginBottom: "24px" }}>
                <Card style={{ width: "100%", maxWidth: "600px" }}>
                    <Search
                        placeholder="Search tags or contributors..."
                        allowClear
                        size="large"
                        prefix={<SearchOutlined />}
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                </Card>
            </div>

            {/* Tags Grid */}
            {filteredTags.length === 0 ? (
                <Empty
                    description={searchTerm ? "No tags found matching your search" : "No tags available"}
                    style={{ margin: "60px 0" }}
                />
            ) : (
                <Row gutter={[16, 16]}>
                    {filteredTags.map((tagInfo) => (
                        <Col xs={24} sm={12} md={8} lg={6} xl={4} xxl={3} key={tagInfo.tag}>
                            <Card
                                hoverable
                                onClick={() => handleTagClick(tagInfo.tag)}
                                style={{ height: "100%" }}
                                styles={{ body: { padding: "16px" } }}
                            >
                                <div style={{ marginBottom: "12px" }}>
                                    <Tag
                                        color={getTagColor(tagInfo.fold_count)}
                                        style={{
                                            fontSize: "14px",
                                            padding: "4px 8px",
                                            fontWeight: "bold",
                                            maxWidth: "100%",
                                            overflow: "hidden",
                                            textOverflow: "ellipsis",
                                            whiteSpace: "nowrap"
                                        }}
                                        title={tagInfo.tag}
                                    >
                                        {tagInfo.tag}
                                    </Tag>
                                </div>

                                <Space direction="vertical" style={{ width: "100%" }} size="small">
                                    <div>
                                        <Text strong>{tagInfo.fold_count}</Text>
                                        <Text type="secondary"> fold{tagInfo.fold_count !== 1 ? 's' : ''}</Text>
                                    </div>

                                    <div>
                                        <Text type="secondary" style={{ fontSize: "12px" }}>
                                            Contributors:
                                        </Text>
                                        <div style={{ marginTop: "4px" }}>
                                            <Avatar.Group max={{ count: 3 }} size="small">
                                                {tagInfo.contributors.map((contributor, index) => (
                                                    <Tooltip title={contributor} key={index}>
                                                        <Avatar
                                                            size="small"
                                                            style={{ backgroundColor: `hsl(${(contributor.charCodeAt(0) * 137.5) % 360}, 50%, 50%)` }}
                                                        >
                                                            {contributor.charAt(0).toUpperCase()}
                                                        </Avatar>
                                                    </Tooltip>
                                                ))}
                                            </Avatar.Group>
                                        </div>
                                    </div>

                                    {tagInfo.recent_folds.length > 0 && (
                                        <div>
                                            <Text type="secondary" style={{ fontSize: "12px" }}>
                                                Recent folds:
                                            </Text>
                                            <div style={{ marginTop: "4px" }}>
                                                {tagInfo.recent_folds.slice(0, 2).map((foldName, index) => (
                                                    <div key={index}>
                                                        <Text
                                                            ellipsis
                                                            style={{
                                                                fontSize: "11px",
                                                                color: "#666",
                                                                display: "block"
                                                            }}
                                                        >
                                                            {foldName}
                                                        </Text>
                                                    </div>
                                                ))}
                                                {tagInfo.recent_folds.length > 2 && (
                                                    <Text type="secondary" style={{ fontSize: "11px" }}>
                                                        +{tagInfo.fold_count - 2} more
                                                    </Text>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </Space>
                            </Card>
                        </Col>
                    ))}
                </Row>
            )}
        </div>
    );
}

export default TagsView;
