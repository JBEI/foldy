import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Table, Button, Modal, Form, Input, Select, Typography, Space, Card, Pagination, Tag, Spin } from 'antd';
import { PlusOutlined, EyeOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons';
import { ColumnsType } from 'antd/es/table';
import { Campaign, Fold as FoldType } from '../types/types';
import { PaginatedCampaignsResponse, getCampaigns, createCampaign, deleteCampaign } from '../api/campaignApi';
import { getFold, getFoldsWithPagination } from '../api/foldApi';
import { notify } from '../services/NotificationService';
import { ESMModelPicker } from './FoldView/ESMModelPicker';
import debounce from 'lodash/debounce';
import type { SelectProps } from 'antd';

const { Title, Text } = Typography;
const { TextArea } = Input;

interface FoldOption {
    label: string;
    value: number;
    owner_email?: string;
}

interface LocalFold {
    id: number;
    name: string;
    owner_email?: string;
}

const CampaignsView: React.FC = () => {
    const navigate = useNavigate();
    const [campaigns, setCampaigns] = useState<Campaign[]>([]);
    const [loading, setLoading] = useState(false);
    const [totalCampaigns, setTotalCampaigns] = useState(0);
    const [currentPage, setCurrentPage] = useState(1);
    const [pageSize] = useState(20);

    const [folds, setFolds] = useState<LocalFold[]>([]);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [createForm] = Form.useForm();

    const loadCampaignsAndFolds = async (page: number = 1) => {
        setLoading(true);
        try {
            // First, load campaigns
            const response: PaginatedCampaignsResponse = await getCampaigns(page, pageSize);
            setCampaigns(response.campaigns);
            setTotalCampaigns(response.total);
            setCurrentPage(response.page);

            // Get unique fold IDs from the campaigns
            const foldIds = [...new Set(response.campaigns.map(campaign => campaign.fold_id))];

            // Load only the relevant folds
            const foldsPromises = foldIds.map(async foldId => {
                try {
                    return await getFold(foldId);
                } catch (error) {
                    console.warn(`Failed to load fold ${foldId}:`, error);
                    return null;
                }
            });
            const foldsResponses = await Promise.all(foldsPromises);

            // Map to the format needed for the select dropdown, filtering out null responses
            const foldsData = foldsResponses
                .filter(fold => fold !== null)
                .map(fold => ({
                    id: fold!.id || 0,
                    name: fold!.name,
                    owner_email: fold!.owner
                }));
            setFolds(foldsData);

        } catch (error) {
            notify.error('Failed to load campaigns and folds');
            console.error('Error loading campaigns and folds:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadCampaignsAndFolds();
    }, []);

    const fetchFoldOptions = async (searchTerm: string): Promise<FoldOption[]> => {
        try {
            // If search term is empty, fetch recent folds by passing empty string or null
            const result = await getFoldsWithPagination(searchTerm.trim() || "", null, 1, 20);
            return result.data.map(fold => ({
                label: fold.name,
                value: fold.id || 0,
                owner_email: fold.owner
            }));
        } catch (error) {
            notify.error('Failed to search folds');
            console.error('Error searching folds:', error);
            return [];
        }
    };

    const FoldDebounceSelect: React.FC<{
        value?: number;
        onChange?: (value: number) => void;
        placeholder?: string;
    }> = ({ value, onChange, placeholder }) => {
        const [fetching, setFetching] = useState(false);
        const [options, setOptions] = useState<FoldOption[]>([]);
        const fetchRef = useRef(0);

        const debounceFetcher = useMemo(() => {
            const loadOptions = (searchValue: string) => {
                fetchRef.current += 1;
                const fetchId = fetchRef.current;
                setOptions([]);
                setFetching(true);

                fetchFoldOptions(searchValue).then((newOptions) => {
                    if (fetchId !== fetchRef.current) {
                        return;
                    }
                    setOptions(newOptions);
                    setFetching(false);
                });
            };
            return debounce(loadOptions, 300);
        }, []);

        // Load recent folds on mount
        useEffect(() => {
            debounceFetcher("");
        }, [debounceFetcher]);

        return (
            <Select
                showSearch
                value={value}
                placeholder={placeholder}
                filterOption={false}
                onSearch={debounceFetcher}
                onChange={onChange}
                notFoundContent={fetching ? <Spin size="small" /> : 'No results found'}
                style={{ width: '100%' }}
            >
                {options.map(option => (
                    <Select.Option key={option.value} value={option.value}>
                        {option.label}{option.owner_email ? ` | ${option.owner_email}` : ''}
                    </Select.Option>
                ))}
            </Select>
        );
    };

    const handleCreateCampaign = async (values: any) => {
        try {
            await createCampaign({
                name: values.name,
                fold_id: values.fold_id,
                description: values.description,
                naturalness_model: values.naturalness_model,
                embedding_model: values.embedding_model,
                domain_boundaries: values.domain_boundaries,
            });
            notify.success('Campaign created successfully');
            setShowCreateModal(false);
            createForm.resetFields();
            loadCampaignsAndFolds(currentPage);
        } catch (error: any) {
            notify.error(error.response?.data?.message || 'Failed to create campaign');
        }
    };

    const handleDeleteCampaign = async (campaignId: number, campaignName: string) => {
        Modal.confirm({
            title: 'Delete Campaign',
            content: `Are you sure you want to delete campaign "${campaignName}"? This action cannot be undone.`,
            okText: 'Delete',
            okType: 'danger',
            onOk: async () => {
                try {
                    await deleteCampaign(campaignId);
                    notify.success('Campaign deleted successfully');
                    loadCampaignsAndFolds(currentPage);
                } catch (error: any) {
                    notify.error(error.response?.data?.message || 'Failed to delete campaign');
                }
            },
        });
    };

    const columns: ColumnsType<Campaign> = [
        {
            title: 'Name',
            dataIndex: 'name',
            key: 'name',
            width: '25%',
            render: (name: string, record: Campaign) => (
                <a
                    href={`/campaigns/${record.id}`}
                    onClick={(e) => {
                        e.preventDefault();
                        navigate(`/campaigns/${record.id}`);
                    }}
                    style={{
                        color: '#1a1a1a',
                        fontSize: '16px',
                        fontWeight: 600,
                        cursor: 'pointer',
                        textDecoration: 'none',
                        transition: 'all 0.2s',
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.textDecoration = 'underline';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.textDecoration = 'none';
                    }}
                >
                    {name}
                </a>
            ),
        },
        {
            title: 'Description',
            dataIndex: 'description',
            key: 'description',
            width: '30%',
            ellipsis: true,
            render: (description: string) => (
                <Text style={{ fontSize: '14px', color: '#262626' }}>
                    {description || <span style={{ color: '#8c8c8c' }}>No description</span>}
                </Text>
            ),
        },
        {
            title: 'Fold',
            dataIndex: 'fold_name',
            key: 'fold_name',
            width: '15%',
            render: (foldName: string, record: Campaign) => (
                <a
                    href={`/fold/${record.fold_id}`}
                    onClick={(e) => {
                        e.preventDefault();
                        navigate(`/fold/${record.fold_id}`);
                    }}
                    style={{
                        color: '#595959',
                        fontSize: '13px',
                        cursor: 'pointer',
                        textDecoration: 'none',
                        transition: 'all 0.2s',
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.color = '#1890ff';
                        e.currentTarget.style.textDecoration = 'underline';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.color = '#595959';
                        e.currentTarget.style.textDecoration = 'none';
                    }}
                >
                    {foldName}
                </a>
            ),
        },
        {
            title: 'Fold Owner',
            key: 'owner',
            width: '12%',
            render: (_, record: Campaign) => {
                const fold = folds.find(f => f.id === record.fold_id);
                return (
                    <Text style={{ color: '#595959', fontSize: '12px' }}>
                        {fold?.owner_email || '-'}
                    </Text>
                );
            },
        },
        {
            title: 'Rounds',
            key: 'rounds',
            width: '8%',
            render: (_, record: Campaign) => (
                <Text style={{ fontSize: '13px' }}>{record.rounds?.length || 0}</Text>
            ),
        },
        {
            title: 'Created',
            dataIndex: 'created_at',
            key: 'created_at',
            width: '10%',
            render: (date: string) => (
                <Text style={{ fontSize: '12px', color: '#595959' }}>
                    {new Date(date).toLocaleDateString()}
                </Text>
            ),
            sorter: (a: Campaign, b: Campaign) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
            defaultSortOrder: 'descend' as const,
        },
        {
            title: 'Actions',
            key: 'actions',
            width: 100,
            fixed: 'right',
            render: (_, record: Campaign) => (
                <Space size="small">
                    <Button
                        type="text"
                        size="small"
                        icon={<EyeOutlined />}
                        onClick={() => navigate(`/campaigns/${record.id}`)}
                        title="View campaign"
                    />
                    <Button
                        type="text"
                        size="small"
                        danger
                        icon={<DeleteOutlined />}
                        onClick={() => handleDeleteCampaign(record.id, record.name)}
                        title="Delete campaign"
                    />
                </Space>
            ),
        },
    ];

    return (
        <div style={{ padding: '24px', height: '100%', overflow: 'auto' }}>
            <Card>
                <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <Title level={2} style={{ margin: 0 }}>Campaigns</Title>
                        <Text type="secondary">Manage your directed evolution campaigns</Text>
                    </div>
                    <Button
                        type="primary"
                        icon={<PlusOutlined />}
                        onClick={() => setShowCreateModal(true)}
                    >
                        New Campaign
                    </Button>
                </div>

                <div style={{
                    overflowX: 'auto',
                    marginBottom: '16px',
                    WebkitOverflowScrolling: 'touch',
                }}>
                    <Table
                        columns={columns}
                        dataSource={campaigns}
                        loading={loading}
                        rowKey="id"
                        pagination={false}
                        style={{ minWidth: '800px' }}
                        size="middle"
                        onRow={() => ({
                            style: {
                                transition: 'background-color 0.2s',
                            },
                            onMouseEnter: (e: React.MouseEvent<HTMLTableRowElement>) => {
                                e.currentTarget.style.backgroundColor = '#fafafa';
                            },
                            onMouseLeave: (e: React.MouseEvent<HTMLTableRowElement>) => {
                                e.currentTarget.style.backgroundColor = '';
                            },
                        })}
                    />
                </div>

                <div style={{ textAlign: 'right' }}>
                    <Pagination
                        current={currentPage}
                        total={totalCampaigns}
                        pageSize={pageSize}
                        onChange={(page) => {
                            setCurrentPage(page);
                            loadCampaignsAndFolds(page);
                        }}
                        showSizeChanger={false}
                        showQuickJumper
                        showTotal={(total, range) =>
                            `${range[0]}-${range[1]} of ${total} campaigns`
                        }
                    />
                </div>
            </Card>

            <Modal
                title="Create New Campaign"
                open={showCreateModal}
                onCancel={() => {
                    setShowCreateModal(false);
                    createForm.resetFields();
                }}
                footer={null}
            >
                <Form
                    form={createForm}
                    layout="vertical"
                    onFinish={handleCreateCampaign}
                >
                    <Form.Item
                        name="name"
                        label="Campaign Name"
                        rules={[{ required: true, message: 'Please enter a campaign name' }]}
                    >
                        <Input placeholder="e.g., High Activity Evolution" />
                    </Form.Item>

                    <Form.Item
                        name="fold_id"
                        label="Fold"
                        rules={[{ required: true, message: 'Please select a fold' }]}
                    >
                        <FoldDebounceSelect
                            placeholder="Search and select a fold for this campaign"
                            value={createForm.getFieldValue('fold_id')}
                            onChange={(value) => createForm.setFieldValue('fold_id', value)}
                        />
                    </Form.Item>

                    <Form.Item
                        name="description"
                        label="Description (Optional)"
                    >
                        <TextArea
                            rows={3}
                            placeholder="Describe the goals and approach for this campaign..."
                        />
                    </Form.Item>

                    <Form.Item
                        name="naturalness_model"
                        label="Naturalness Protein Language Model"
                        initialValue="esm2_t30_150M_UR50D"
                    >
                        <ESMModelPicker
                            value={createForm.getFieldValue('naturalness_model') || "esm2_t30_150M_UR50D"}
                            onChange={(value) => createForm.setFieldValue('naturalness_model', value)}
                            label=""
                        />
                    </Form.Item>

                    <Form.Item
                        name="embedding_model"
                        label="Embedding Model"
                        initialValue="esm2_t30_150M_UR50D"
                    >
                        <ESMModelPicker
                            value={createForm.getFieldValue('embedding_model') || "esm2_t30_150M_UR50D"}
                            onChange={(value) => createForm.setFieldValue('embedding_model', value)}
                            label=""
                        />
                    </Form.Item>

                    <Form.Item
                        name="domain_boundaries"
                        label="Domain Boundaries"
                        tooltip="Optional comma-separated list of boundaries for domain-pooling when generating embeddings (e.g., 10,50,100)"
                    >
                        <Input placeholder="e.g., 10,50,100" />
                    </Form.Item>

                    <Form.Item style={{ marginTop: '24px', marginBottom: 0 }}>
                        <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
                            <Button onClick={() => {
                                setShowCreateModal(false);
                                createForm.resetFields();
                            }}>
                                Cancel
                            </Button>
                            <Button type="primary" htmlType="submit">
                                Create Campaign
                            </Button>
                        </Space>
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
};

export default CampaignsView;
