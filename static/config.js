const HOST = '';

function apiHost() {
  if (location.search.includes('use_local_host')) {
    return 'http://localhost:8000';
  }
  const hostInStorage = localStorage.getItem('chainer-flowerizer-server')
  if (hostInStorage !== null) {
    return hostInStorage;
  }
  return HOST;
}

const config = {
  apiHost: apiHost(),
};
