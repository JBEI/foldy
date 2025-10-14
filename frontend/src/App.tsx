import React, { lazy, Suspense, useEffect, useState, useRef } from "react";
import { useJwt } from "react-jwt";
import {
    BrowserRouter,
    Route,
    Routes,
    useLocation,
    useNavigate,
    useSearchParams,
    Link,
} from "react-router-dom";
import "react-tiny-fab/dist/styles.css";
import "./App.scss";

import { GoogleOAuthProvider } from "@react-oauth/google";
import UIkit from "uikit";
import { Layout, Menu, Button as AntButton, Drawer, Spin, Tour, Modal, Typography as AntTypography, ConfigProvider, App as AntApp } from "antd";
import { MenuOutlined, HomeOutlined, InfoCircleOutlined, SettingOutlined, DatabaseOutlined, TagOutlined, ExperimentOutlined, QuestionCircleOutlined, RocketOutlined } from "@ant-design/icons";
import About from "./components/AboutView/About";
import DashboardView from "./components/DashboardView";
import NewBoltzFoldView from "./components/NewFoldView/NewBoltzFoldView";
// import NewFold from "./components/NewFoldView/NewFold2Uniforms";
// import NewFold from "./components/NewFoldView/NewFoldView";
// import NewFold from "./components/NewFoldView/NewFold";
import SudoPage from "./components/SudoPageView/SudoPage";
import CampaignsView from "./components/CampaignsView";
import CampaignView from "./components/CampaignView";
import {
    authenticationService,
    currentJwtStringSubject,
    DecodedJwt,
    getDescriptionOfUserType,
    isFullDecodedJwt,
    LoginButton,
    UserProfileDropdown,
} from "./services/authentication.service";
import TagView from "./TagView";
import TagsView from "./components/TagsView";
import { FoldingAtTheDisco, FoldyMascot } from "./util/foldyMascot";
import { notify } from "./services/NotificationService";
import { useKeyboardIntercept } from "./util/keyboardInterceptor";
import { getMessages } from "./api/adminApi";

const AvatarFoldView = lazy(() => import("./components/FoldView/FoldView"));

function CheckForErrorQueryString() {
    const location = useLocation();
    const navigate = useNavigate();
    let params = new URLSearchParams(location.search);

    const queryParamErrorText = params.get("error_message");
    if (!queryParamErrorText) {
        return <div></div>;
    }

    notify.error(queryParamErrorText);

    params.delete("error_message");
    navigate({
        pathname: location.pathname,
        search: params.toString(),
    });

    return <div></div>;
}

interface NavLinkProps {
    href: string;
    children: React.ReactNode;
    external?: boolean;
}

function NavLink({ href, children, external = false }: NavLinkProps) {
    const commonStyles = {
        color: "#fff",
        textDecoration: 'none' as const,
        padding: '6px 8px',
        borderRadius: '4px',
        transition: 'background-color 0.2s',
        fontSize: '15px',
    };

    const handleMouseEnter = (e: React.MouseEvent<HTMLElement>) => {
        (e.target as HTMLElement).style.backgroundColor = 'rgba(255,255,255,0.1)';
    };

    const handleMouseLeave = (e: React.MouseEvent<HTMLElement>) => {
        (e.target as HTMLElement).style.backgroundColor = 'transparent';
    };

    if (external) {
        return (
            <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                style={commonStyles}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
            >
                {children}
            </a>
        );
    }

    return (
        <Link
            to={href}
            style={commonStyles}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
        >
            {children}
        </Link>
    );
}

function RoutedApp({ token, setToken }: {
    token: string | null;
    setToken: React.Dispatch<React.SetStateAction<string | null>>;
}) {
    const { decodedToken, isExpired } = useJwt(token || '');
    let [searchParams, setSearchParams] = useSearchParams();
    const [cartwheelingMascotList, setCartwheelingMascotList] = useState<React.ReactElement[]>([]);
    const [enableDisco, setEnableDisco] = useState(false);
    const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
    const [isMobile, setIsMobile] = useState(window.innerWidth < 960);
    const [tourOpen, setTourOpen] = useState(false);
    const [newUserModalOpen, setNewUserModalOpen] = useState(false);
    const navigate = useNavigate();
    const location = useLocation();

    // Tour refs
    const dashboardNavRef = useRef(null);
    const viewAllRef = useRef(null);
    const newFoldRef = useRef(null);
    const campaignsNavRef = useRef(null);

    // Tour steps
    const tourSteps = [
        {
            title: 'Welcome to the Dashboard!',
            description: 'This is where you can view predicted protein structures, or folds.',
            target: () => dashboardNavRef.current,
        },
        {
            title: 'View All Folds',
            description: 'You can use the search bar to search for folds by name or user, or click View All to see all predicted structures.',
            target: () => viewAllRef.current,
        },
        {
            title: 'Create New Folds',
            description: 'Click this button to predict the structure of a protein or complex with Boltz-2. Most tools are linked to your created folds.',
            target: () => newFoldRef.current,
        },
        {
            title: 'Protein Engineering Campaigns',
            description: 'You can run protein engineering campaigns here. First fold your protein, then head over to Campaigns to start an engineering campaign!',
            target: () => campaignsNavRef.current,
        },
    ];

    // Start walkthrough handler
    const startWalkthrough = () => {
        // Navigate to dashboard first
        if (location.pathname !== '/') {
            navigate('/');
        }
        // Small delay to ensure navigation completes
        setTimeout(() => {
            setTourOpen(true);
        }, 100);
    };

    var fullDecodedToken: DecodedJwt | null = null;
    if (isFullDecodedJwt(decodedToken)) {
        fullDecodedToken = decodedToken;
    }

    // Handle new user modal in useEffect to avoid infinite renders
    useEffect(() => {
        if (fullDecodedToken) {
            const isNewUser = searchParams.get("new_user");
            if (isNewUser) {
                const newSearchParams = new URLSearchParams(searchParams);
                newSearchParams.delete("new_user");
                setSearchParams(newSearchParams);
                setNewUserModalOpen(true);
            }
        }
    }, [fullDecodedToken, searchParams, setSearchParams]);

    useKeyboardIntercept('f', () => {
        setCartwheelingMascotList([...cartwheelingMascotList, <FoldyMascot text={""} moveTextAbove={false} isCartwheeling={true} key={cartwheelingMascotList.length} isKanKaning={false} />]);
    });

    useKeyboardIntercept('k', () => {
        setCartwheelingMascotList([...cartwheelingMascotList, <FoldyMascot text={""} moveTextAbove={false} isCartwheeling={true} key={cartwheelingMascotList.length} isKanKaning={true} />]);
    });

    useKeyboardIntercept('d', () => {
        setEnableDisco(!enableDisco);
    });

    useEffect(() => {
        const handleResize = () => {
            setIsMobile(window.innerWidth < 960);
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // Check for system messages on app load
    useEffect(() => {
        const checkMessages = async () => {
            try {
                const messages = await getMessages();
                messages.forEach((msg) => {
                    if (msg.type === 'warning') {
                        // Use timeout: 0 to make the notification persistent
                        notify.show({ message: msg.message, type: 'warning', timeout: 0 });
                    }
                });
            } catch (error) {
                console.error('Failed to fetch system messages:', error);
            }
        };
        checkMessages();
    }, []);

    const renderLoader = () => {
        return (
            <div style={{ textAlign: 'center', padding: '60px 0' }}>
                <Spin size="large" />
            </div>
        );
    };

    const foldyTitle = (
        <span>
            {import.meta.env.VITE_INSTITUTION} Foldy
            <sub>
                <sub>
                    {fullDecodedToken?.user_claims.type === "viewer" ? "View Only" : null}
                    {fullDecodedToken?.user_claims.type === "editor"
                        ? "Edit Access"
                        : null}
                    {fullDecodedToken?.user_claims.type === "admin"
                        ? "Admin Access"
                        : null}
                </sub>
            </sub>
        </span>
    );

    const foldyWelcomeText = `Welcome to ${import.meta.env.VITE_INSTITUTION} Foldy! Login with an ${import.meta.env.VITE_INSTITUTION} account for edit access, or any other account to view public structures.`;

    const menuItems = [
        {
            key: 'dashboard',
            icon: <HomeOutlined />,
            label: 'Dashboard',
            onClick: () => navigate('/')
        },
        {
            key: 'campaigns',
            icon: <ExperimentOutlined />,
            label: 'Campaigns',
            onClick: () => navigate('/campaigns')
        },
        {
            key: 'tags',
            icon: <TagOutlined />,
            label: 'Tags',
            onClick: () => navigate('/tags')
        },
        ...(fullDecodedToken?.user_claims.type === "admin" ? [
            {
                key: 'rq',
                icon: <SettingOutlined />,
                label: 'RQ',
                onClick: () => window.open(`${import.meta.env.VITE_BACKEND_URL}/rq/`, '_blank')
            },
            {
                key: 'dbs',
                icon: <DatabaseOutlined />,
                label: 'DBs',
                onClick: () => window.open(`${import.meta.env.VITE_BACKEND_URL}/admin/`, '_blank')
            },
            {
                key: 'sudo',
                icon: <SettingOutlined />,
                label: 'Sudo Page',
                onClick: () => navigate('/sudopage')
            }
        ] : []),
        {
            key: 'about',
            icon: <InfoCircleOutlined />,
            label: 'About',
            onClick: () => navigate('/about')
        }
    ];

    const desktop_navbar = (
        <Layout.Header
            style={{
                background: "linear-gradient(to left, #1e87f0, #1565c0)",
                padding: '0 20px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                lineHeight: 'normal',
                height: '48px',
                minHeight: '48px'
            }}
        >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                <Link
                    to="/"
                    style={{
                        color: "#fff",
                        textDecoration: 'none',
                        fontSize: '18px',
                        whiteSpace: 'nowrap'
                    }}
                >
                    {foldyTitle}
                </Link>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                    <span ref={dashboardNavRef}>
                        <NavLink href="/">Dashboard</NavLink>
                    </span>
                    <span ref={campaignsNavRef}>
                        <NavLink href="/campaigns">Campaigns</NavLink>
                    </span>
                    <NavLink href="/tags">Tags</NavLink>
                    {fullDecodedToken?.user_claims.type === "admin" && (
                        <>
                            <NavLink href={`${import.meta.env.VITE_BACKEND_URL}/rq/`} external>RQ</NavLink>
                            <NavLink href={`${import.meta.env.VITE_BACKEND_URL}/admin/`} external>DBs</NavLink>
                            <NavLink href="/sudopage">Sudo Page</NavLink>
                        </>
                    )}
                    <NavLink href="/about">About</NavLink>
                </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                {fullDecodedToken && !isExpired && (
                    <AntButton
                        type="text"
                        icon={<QuestionCircleOutlined />}
                        onClick={startWalkthrough}
                        style={{
                            color: '#fff',
                            border: 'none',
                            fontSize: '14px'
                        }}
                    >
                        Walkthrough
                    </AntButton>
                )}
                <UserProfileDropdown
                    decodedToken={fullDecodedToken}
                    setToken={setToken}
                    isExpired={isExpired}
                />
                {fullDecodedToken && !isExpired ? null : (
                    location.pathname === '/' ? null : (
                        <FoldyMascot text={foldyWelcomeText} moveTextAbove={false} isCartwheeling={false} isKanKaning={false} />
                    )
                )}
            </div>
        </Layout.Header>
    );

    const mobile_navbar = (
        <Layout.Header
            style={{
                background: "linear-gradient(to left, #1e87f0, #1565c0)",
                zIndex: 100,
                position: "fixed",
                top: 0,
                width: "100%",
                padding: '0',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                height: '48px',
                minHeight: '48px'
            }}
        >
            <Link
                to="/"
                style={{
                    color: "#fff",
                    textDecoration: 'none',
                    fontSize: '20px',
                    flex: '1',
                    paddingLeft: '16px'
                }}
            >
                {foldyTitle}
            </Link>

            <AntButton
                type="text"
                icon={<MenuOutlined />}
                onClick={() => setMobileDrawerOpen(true)}
                style={{
                    color: '#fff',
                    border: 'none',
                    padding: '4px 8px',
                    minWidth: '40px',
                    height: '40px',
                    marginRight: '16px'
                }}
            />

            {fullDecodedToken && !isExpired ? null : (
                location.pathname === '/' ? null : (
                    <FoldyMascot text={foldyWelcomeText} moveTextAbove={true} isCartwheeling={false} isKanKaning={false} />
                )
            )}
        </Layout.Header>
    );

    return (
        <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
            <div style={{ display: isMobile ? 'none' : 'block' }}>{desktop_navbar}</div>
            <div style={{ display: isMobile ? 'block' : 'none', paddingTop: "80px" }}>{mobile_navbar}</div>

            <CheckForErrorQueryString />

            <Drawer
                title={`${import.meta.env.VITE_INSTITUTION} Foldy`}
                placement="right"
                onClose={() => setMobileDrawerOpen(false)}
                open={mobileDrawerOpen}
                styles={{
                    body: { padding: '24px 0' }
                }}
            >
                <div style={{ marginBottom: '24px', padding: '0 24px' }}>
                    <p>
                        {import.meta.env.VITE_INSTITUTION} Foldy is a web app for
                        predicting and using protein structures based on AlphaFold.
                    </p>
                </div>

                <Menu
                    mode="vertical"
                    items={menuItems}
                    onClick={() => setMobileDrawerOpen(false)}
                    style={{ border: 'none' }}
                />

                <div style={{ marginTop: '24px', padding: '0 24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    {fullDecodedToken && !isExpired && (
                        <AntButton
                            type="default"
                            icon={<QuestionCircleOutlined />}
                            onClick={() => {
                                setMobileDrawerOpen(false);
                                startWalkthrough();
                            }}
                            style={{ width: '100%' }}
                        >
                            Walkthrough
                        </AntButton>
                    )}
                    <LoginButton
                        setToken={setToken}
                        decodedToken={fullDecodedToken}
                        isExpired={isExpired}
                    />
                </div>
            </Drawer>

            <div
                className={location.pathname.startsWith('/fold/') ?
                    "uk-container-expand" :
                    "uk-width-5-6@xl uk-container-center uk-align-center"
                }
                style={{
                    display: "flex",
                    flexDirection: "column",
                    flexGrow: 1,
                    overflow: "hidden",
                    marginTop: "0px",
                    marginBottom: "0px",
                }}
            >
                <Routes>
                    <Route
                        path="/fold/:foldId"
                        element={
                            <Suspense fallback={renderLoader()}>
                                <AvatarFoldView
                                    userType={
                                        fullDecodedToken ? fullDecodedToken.user_claims.type : null
                                    }
                                />
                            </Suspense>
                        }
                    />
                    <Route
                        path="/fold/:foldId/:tabName"
                        element={
                            <Suspense fallback={renderLoader()}>
                                <AvatarFoldView
                                    userType={
                                        fullDecodedToken ? fullDecodedToken.user_claims.type : null
                                    }
                                />
                            </Suspense>
                        }
                    />
                    <Route
                        path="/campaigns"
                        element={<CampaignsView />}
                    />
                    <Route
                        path="/campaigns/:campaignId"
                        element={<CampaignView />}
                    />
                    <Route
                        path="/campaigns/:campaignId/:roundNumber"
                        element={<CampaignView />}
                    />
                    <Route
                        path="/campaigns/:campaignId/:roundNumber/:subpage"
                        element={<CampaignView />}
                    />
                    <Route
                        path="/tag/:tagStringParam"
                        element={<TagView />}
                    />
                    <Route
                        path="/tags"
                        element={<TagsView />}
                    />
                    <Route
                        path="/newFold"
                        element={
                            <NewBoltzFoldView
                                userType={
                                    fullDecodedToken ? fullDecodedToken.user_claims.type : null
                                }
                            />
                        }
                    />
                    <Route
                        path="/sudopage"
                        element={<SudoPage />}
                    />
                    <Route
                        path="/about"
                        element={
                            <About
                                userType={
                                    fullDecodedToken ? fullDecodedToken.user_claims.type : null
                                }
                            />
                        }
                    />
                    <Route
                        path="/"
                        element={
                            <DashboardView
                                decodedToken={fullDecodedToken}
                                viewAllRef={viewAllRef}
                                newFoldRef={newFoldRef}
                            />
                        }
                    />
                </Routes>
            </div>
            {cartwheelingMascotList.length > 0 ? cartwheelingMascotList : null}
            <FoldingAtTheDisco enabled={enableDisco} />

            {/* Tour component */}
            <Tour
                open={tourOpen}
                onClose={() => setTourOpen(false)}
                steps={tourSteps}
                type="primary"
            />

            {/* New User Welcome Modal */}
            <Modal
                title={
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <RocketOutlined style={{ color: '#1890ff' }} />
                        <span>🎉 Welcome to {import.meta.env.VITE_INSTITUTION} Foldy!</span>
                    </div>
                }
                open={newUserModalOpen}
                onOk={() => {
                    setNewUserModalOpen(false);
                    startWalkthrough();
                }}
                onCancel={() => {
                    setNewUserModalOpen(false);
                }}
                okText="🚀 Start Interactive Walkthrough"
                cancelText="Skip and Explore"
                width={600}
                centered
            >
                <div style={{ padding: '8px 0' }}>
                    <AntTypography.Paragraph>
                        <strong>Your access level:</strong> {fullDecodedToken ? getDescriptionOfUserType(
                            fullDecodedToken.user_claims.type || ""
                        ) : ''}
                    </AntTypography.Paragraph>

                    <AntTypography.Title level={4} style={{ marginTop: '24px' }}>🧬 What is Foldy?</AntTypography.Title>
                    <AntTypography.Paragraph>
                        Foldy is a democratized protein folding platform that uses cutting-edge AI models (like Boltz-2x)
                        to predict protein structures with exceptional accuracy for complex scenarios including multimers,
                        small molecule docking, and nucleic acid interactions.
                    </AntTypography.Paragraph>

                    <AntTypography.Title level={4} style={{ marginTop: '24px' }}>🚀 Get Started:</AntTypography.Title>
                    <ul style={{ paddingLeft: '20px' }}>
                        <li>Browse existing structures from the <strong>Dashboard</strong></li>
                        <li>Create predictions by clicking <strong>"New Fold"</strong> (editors only)</li>
                        <li>Explore the comprehensive analysis tools in each fold</li>
                        <li>Run protein engineering <strong>Campaigns</strong> for optimization</li>
                    </ul>

                    <div style={{
                        backgroundColor: '#f6ffed',
                        border: '1px solid #b7eb8f',
                        borderRadius: '4px',
                        padding: '12px',
                        margin: '16px 0'
                    }}>
                        <AntTypography.Title level={5} style={{ color: '#389e0d', marginTop: 0 }}>
                            📚 Citations & Attribution
                        </AntTypography.Title>
                        <AntTypography.Paragraph style={{ marginBottom: '8px', color: '#389e0d' }}>
                            If you publish research using this platform, please consider citing the relevant papers
                            to support the developers. This includes both the Foldy platform and underlying methods like Boltz-2x.
                        </AntTypography.Paragraph>
                    </div>

                    <AntTypography.Paragraph style={{ marginBottom: 0 }}>
                        Visit the <a href="/about">About page</a> for detailed information, FAQs, and complete citation requirements.
                    </AntTypography.Paragraph>
                </div>
            </Modal>
        </div>
    );
}


// -----------------------------------------------
// 1) Make a tiny "Bootstrapping" or "InitApp" component
//    that sets the token from the URL, then calls setInitDone.
// -----------------------------------------------
function InitApp({
    onInitDone,
}: {
    onInitDone: (token: string | null) => void;
}) {
    const [searchParams, setSearchParams] = useSearchParams();

    useEffect(() => {
        // Look for the token in the URL
        const jwtString = searchParams.get("access_token");
        if (jwtString) {
            // Remove access_token from the URL
            const newSearchParams = new URLSearchParams(searchParams);
            newSearchParams.delete("access_token");
            setSearchParams(newSearchParams);

            // Persist the token in localStorage
            localStorage.setItem("currentJwtString", jwtString);
            currentJwtStringSubject.next(jwtString);

            // Let parent know initialization is done
            console.log(`IN INITAPP, calling onInitDone with token: ${jwtString}`);
            onInitDone(jwtString);
        } else {
            // No token in URL; see if there's one in localStorage
            const existingToken = localStorage.getItem("currentJwtString");
            console.log(`IN INITAPP, calling onInitDone with existing token: ${existingToken}`);
            onInitDone(existingToken);
        }
    }, [searchParams]);

    // While we parse the URL and store the token, show a spinner/loader
    return (
        <div style={{ textAlign: 'center', padding: '60px 0' }}>
            <Spin size="large" />
        </div>
    );
}

function App() {
    const [token, setToken] = useState<string | null>(null);
    const [initDone, setInitDone] = useState(false);

    console.log(`IN APP, initDone: ${initDone}`);

    function handleInitDone(tokenFromUrl: string | null) {
        console.log(`IN APP, calling HandleInitDone with token: ${tokenFromUrl}`);
        setToken(tokenFromUrl);
        setInitDone(true);
    }

    if (!import.meta.env.VITE_INSTITUTION) {
        console.error("VITE_INSTITUTION is unset.");
    }
    if (!import.meta.env.VITE_BACKEND_URL) {
        console.error("VITE_BACKEND_URL is unset.");
    }
    if (!import.meta.env.VITE_GOOGLE_CLIENT_ID) {
        console.error("VITE_GOOGLE_CLIENT_ID is unset.");
    }
    return (
        <ConfigProvider
            theme={{
                token: {
                    colorPrimary: '#1e87f0',
                    colorInfo: '#1e87f0',
                },
            }}
        >
            <GoogleOAuthProvider
                clientId={import.meta.env.VITE_GOOGLE_CLIENT_ID || ""}
            >
                <BrowserRouter>
                    {initDone ? (
                        <RoutedApp token={token} setToken={setToken} />
                    ) : (
                        <InitApp onInitDone={handleInitDone} />
                    )}
                </BrowserRouter>
            </GoogleOAuthProvider>
        </ConfigProvider>
    );
}

export default App;
