#!/bin/bash

export BOOST_ROOT=${HOME}/${BOOST_BASENAME}

if [ ! -e ${BOOST_ROOT}/boost/config.hpp ]; then
    pushd ${HOME}
    wget https://downloads.sourceforge.net/project/boost/boost/1.62.0/${BOOST_BASENAME}.tar.bz2
    rm -rf $BOOST_BASENAME
    tar xf ${BOOST_BASENAME}.tar.bz2
    (cd $BOOST_BASENAME && ./bootstrap.sh && ./b2)
    popd
fi

