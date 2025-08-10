'''
Mohammad Kadkhodaei
1404-05-18, 
'''
class Infrastructure:
    def __init__(self):
    
        '''
        Sample: [{'remaining_vcpu': inf, 'remaining_mem': inf}]
        '''
        self.cloud_nodes = [{'remaining_vcpu': CLOUD_vCPU, 'remaining_mem': CLOUD_MEM} for _ in range(NUM_CLOUD_NODES)]
        '''
        Sample:
        [{'remaining_vcpu': 4, 'remaining_mem': 4}, {'remaining_vcpu': 4, 'remaining_mem': 4}, {'remaining_vcpu': 4, 'remaining_mem': 4}, {'remaining_vcpu': 4, 'remaining_mem': 4}, {'remaining_vcpu': 4, 'remaining_mem': 4}, {'remaining_vcpu': 4, 'remaining_mem': 4}, {'remaining_vcpu': 4, 'remaining_mem': 4}]
        '''
        self.fog_nodes = [{'remaining_vcpu': FOG_vCPU, 'remaining_mem': FOG_MEM} for _ in range(NUM_FOG_NODES)]
        self.edge_nodes = [{'remaining_vcpu': EDGE_vCPU, 'remaining_mem': EDGE_MEM} for _ in range(NUM_EDGE_NODES)]
        
        '''
        1404-05-19
        Sample: {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9], 5: [10, 11], 6: [12, 13]}
        '''
        # Assign edge nodes to fog nodes (2 edges per fog)
        self.fog_edge_mapping = {i: [2*i, 2*i+1] for i in range(NUM_FOG_NODES)}
        
    '''
    Sample: 7+14+1 = 22 and 22*2(cpu,mem) = 44 
    [inf, inf, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    '''
    def get_feature_vector(self):
        """Returns the current state of infrastructure resources"""
        cloud_vcpu = [n['remaining_vcpu'] for n in self.cloud_nodes]
        cloud_mem = [n['remaining_mem'] for n in self.cloud_nodes]
        fog_vcpu = [n['remaining_vcpu'] for n in self.fog_nodes]
        fog_mem = [n['remaining_mem'] for n in self.fog_nodes]
        edge_vcpu = [n['remaining_vcpu'] for n in self.edge_nodes]
        edge_mem = [n['remaining_mem'] for n in self.edge_nodes]
        
        return cloud_vcpu + cloud_mem + fog_vcpu + fog_mem + edge_vcpu + edge_mem
