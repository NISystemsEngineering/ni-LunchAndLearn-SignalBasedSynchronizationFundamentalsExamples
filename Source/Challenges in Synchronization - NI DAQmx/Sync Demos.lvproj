<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="23008000">
	<Property Name="NI.LV.All.SaveVersion" Type="Str">23.0</Property>
	<Property Name="NI.LV.All.SourceOnly" Type="Bool">true</Property>
	<Item Name="My Computer" Type="My Computer">
		<Property Name="IOScan.Faults" Type="Str"></Property>
		<Property Name="IOScan.NetVarPeriod" Type="UInt">100</Property>
		<Property Name="IOScan.NetWatchdogEnabled" Type="Bool">false</Property>
		<Property Name="IOScan.Period" Type="UInt">10000</Property>
		<Property Name="IOScan.PowerupMode" Type="UInt">0</Property>
		<Property Name="IOScan.Priority" Type="UInt">9</Property>
		<Property Name="IOScan.ReportModeConflict" Type="Bool">true</Property>
		<Property Name="IOScan.StartEngineOnDeploy" Type="Bool">false</Property>
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="SubVIs" Type="Folder">
			<Item Name="Find Delta Samples of Rising Edge.vi" Type="VI" URL="../Top Level Demos/Find Delta Samples of Rising Edge.vi"/>
		</Item>
		<Item Name="Top Level Demos" Type="Folder">
			<Item Name="1 Sync - Software Trigger.vi" Type="VI" URL="../Top Level Demos/1 Sync - Software Trigger.vi"/>
			<Item Name="2 Sync - Digital Trigger.vi" Type="VI" URL="../Top Level Demos/2 Sync - Digital Trigger.vi"/>
			<Item Name="3 Sync - Digital Trigger Showing Drift.vi" Type="VI" URL="../Top Level Demos/3 Sync - Digital Trigger Showing Drift.vi"/>
			<Item Name="4 Sync - Shared Time Base No Drift.vi" Type="VI" URL="../Top Level Demos/4 Sync - Shared Time Base No Drift.vi"/>
			<Item Name="5 Sync - TSN Magicness.vi" Type="VI" URL="../Top Level Demos/5 Sync - TSN Magicness.vi"/>
		</Item>
		<Item Name="Dependencies" Type="Dependencies"/>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
