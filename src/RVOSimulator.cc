/*
 * RVOSimulator.cc
 * RVO2 Library
 *
 * SPDX-FileCopyrightText: 2008 University of North Carolina at Chapel Hill
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <https://gamma.cs.unc.edu/RVO2/>
 */

/**
 * @file  RVOSimulator.cc
 * @brief Defines the RVOSimulator class.
 */

#include "RVOSimulator.h"

#include <limits>
#include <utility>

#include "Agent.h"
#include "KdTree.h"
#include "Line.h"
#include "Obstacle.h"
#include "Vector2.h"

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

namespace RVO {
const std::size_t RVO_ERROR = std::numeric_limits<std::size_t>::max();

static Vector2 g_InValidVec2{0, 0};

RVOSimulator::RVOSimulator()
    : defaultAgent_(NULL),
      kdTree_(new KdTree(this)),
      globalTime_(0.0F),
      timeStep_(0.0F),
      totalID_(0),
      bDirty(false) {}


RVOSimulator::RVOSimulator(float timeStep, float neighborDist,
                           std::size_t maxNeighbors, float timeHorizon,
                           float timeHorizonObst, float radius, float maxSpeed)
    : defaultAgent_(new Agent()),
      kdTree_(new KdTree(this)),
      globalTime_(0.0F),
      timeStep_(timeStep),
      totalID_(0) {
  defaultAgent_->maxNeighbors_ = maxNeighbors;
  defaultAgent_->maxSpeed_ = maxSpeed;
  defaultAgent_->neighborDist_ = neighborDist;
  defaultAgent_->radius_ = radius;
  defaultAgent_->timeHorizon_ = timeHorizon;
  defaultAgent_->timeHorizonObst_ = timeHorizonObst;
}

RVOSimulator::RVOSimulator(float timeStep, float neighborDist,
                           std::size_t maxNeighbors, float timeHorizon,
                           float timeHorizonObst, float radius, float maxSpeed,
                           const Vector2 &velocity)
    : defaultAgent_(new Agent()),
      kdTree_(new KdTree(this)),
      globalTime_(0.0F),
      timeStep_(timeStep),
      totalID_(0) {
  defaultAgent_->velocity_ = velocity;
  defaultAgent_->maxNeighbors_ = maxNeighbors;
  defaultAgent_->maxSpeed_ = maxSpeed;
  defaultAgent_->neighborDist_ = neighborDist;
  defaultAgent_->radius_ = radius;
  defaultAgent_->timeHorizon_ = timeHorizon;
  defaultAgent_->timeHorizonObst_ = timeHorizonObst;
}

RVOSimulator::~RVOSimulator() {
//  delete defaultAgent_;
//  delete kdTree_;
//
//  for (std::size_t i = 0U; i < agents_.size(); ++i) {
//    delete agents_[i];
//  }
//
//  for (std::size_t i = 0U; i < obstacles_.size(); ++i) {
//    delete obstacles_[i];
//  }
}

std::size_t RVOSimulator::addAgent(const Vector2 &position) {
  if (defaultAgent_ != NULL) {
    Agent *const agent = new Agent();
    agent->position_ = position;
    agent->velocity_ = defaultAgent_->velocity_;
    agent->id_ = ++totalID_;
    agent->maxNeighbors_ = defaultAgent_->maxNeighbors_;
    agent->maxSpeed_ = defaultAgent_->maxSpeed_;
    agent->neighborDist_ = defaultAgent_->neighborDist_;
    agent->radius_ = defaultAgent_->radius_;
    agent->timeHorizon_ = defaultAgent_->timeHorizon_;
    agent->timeHorizonObst_ = defaultAgent_->timeHorizonObst_;

    {
      std::lock_guard<std::mutex> lock(agentMutex_);
      agents_.emplace(agent->id_, agent);
    }

    bDirty = true;
    return agent->id_;
  }

  return RVO_ERROR;
}

std::size_t RVOSimulator::addAgent(const Vector2 &position, float neighborDist,
                                   std::size_t maxNeighbors, float timeHorizon,
                                   float timeHorizonObst, float radius,
                                   float maxSpeed) {
  return addAgent(position, neighborDist, maxNeighbors, timeHorizon,
                  timeHorizonObst, radius, maxSpeed, Vector2());
}

std::size_t RVOSimulator::addAgent(const Vector2 &position, float neighborDist,
                                   std::size_t maxNeighbors, float timeHorizon,
                                   float timeHorizonObst, float radius,
                                   float maxSpeed, const Vector2 &velocity) {
  Agent *const agent = new Agent();
  agent->position_ = position;
  agent->velocity_ = velocity;
  agent->id_ = ++totalID_;
  agent->maxNeighbors_ = maxNeighbors;
  agent->maxSpeed_ = maxSpeed;
  agent->neighborDist_ = neighborDist;
  agent->radius_ = radius;
  agent->timeHorizon_ = timeHorizon;
  agent->timeHorizonObst_ = timeHorizonObst;
  {
    std::lock_guard<std::mutex> lock(agentMutex_);
    agents_.emplace(agent->id_, agent);
  }

  bDirty = true;
  return agent->id_;
}

void RVOSimulator::delAgetent(std::size_t agentNo) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    it->second->needDelete_ = true;
    bDirty = true;
  }
}

void RVOSimulator::updateDeleteAgent(std::vector<Agent *>  &tempAgentVec, std::vector<size_t> &delAgentVec) {
//    agents_.for_each_m([&](phmap::parallel_flat_hash_map<std::size_t, Agent*>::value_type &pair) {
//        if (pair.second->needDelete_) {
//            delAgentVec.emplace_back(pair.first);
//            bDirty = true;
//        } else {
//            tempAgentVec.emplace_back(pair.second);
//        }
//    });
    std::lock_guard<std::mutex> lock(agentMutex_);
    for (auto it = agents_.begin(); it != agents_.end();) {
      if (it->second->needDelete_) {
        delete it->second;
        bDirty = true;
        it = agents_.erase(it);
      } else {
        tempAgentVec.emplace_back(it->second);
        ++it;
      }
    }
}

std::size_t RVOSimulator::addObstacle(const std::vector<Vector2> &vertices) {
  if (vertices.size() > 1U) {
    const std::size_t obstacleNo = obstacles_.size();

    for (std::size_t i = 0U; i < vertices.size(); ++i) {
      Obstacle *const obstacle = new Obstacle();
      obstacle->point_ = vertices[i];

      if (i != 0U) {
        obstacle->previous_ = obstacles_.back();
        obstacle->previous_->next_ = obstacle;
      }

      if (i == vertices.size() - 1U) {
        obstacle->next_ = obstacles_[obstacleNo];
        obstacle->next_->previous_ = obstacle;
      }

      obstacle->direction_ = normalize(
          vertices[(i == vertices.size() - 1U ? 0U : i + 1U)] - vertices[i]);

      if (vertices.size() == 2U) {
        obstacle->isConvex_ = true;
      } else {
        obstacle->isConvex_ =
            leftOf(vertices[i == 0U ? vertices.size() - 1U : i - 1U],
                   vertices[i],
                   vertices[i == vertices.size() - 1U ? 0U : i + 1U]) >= 0.0F;
      }

      obstacle->id_ = obstacles_.size();

      obstacles_.push_back(obstacle);
    }

    return obstacleNo;
  }

  return RVO_ERROR;
}

void RVOSimulator::doStep() {

  // delete agents_ needDelete_ is true
  std::vector<Agent *> tempAgentVec;
  std::vector<std::size_t> delAgentVec;
  updateDeleteAgent(tempAgentVec, delAgentVec);

  kdTree_->buildAgentTree(tempAgentVec);

#ifdef _OPENMP
//#pragma omp parallel for
#endif /* _OPENMP */
  for (int i = 0; i < static_cast<int>(tempAgentVec.size()); ++i) {
    tempAgentVec[i]->computeNeighbors(kdTree_);
    tempAgentVec[i]->computeNewVelocity(timeStep_);
  }

#ifdef _OPENMP
//#pragma omp parallel for
#endif /* _OPENMP */
  for (int i = 0; i < static_cast<int>(tempAgentVec.size()); ++i) {
    tempAgentVec[i]->update(timeStep_);
  }

//  for (auto it = delAgentVec.begin(); it != delAgentVec.end(); ++it) {
//        auto pair = agents_.find(*it);
//        if (pair != agents_.end()) {
//            delete pair->second;
//            agents_.erase(*it);
//        }
//  }

  globalTime_ += timeStep_;
  bDirty = false;
}

std::size_t RVOSimulator::getAgentAgentNeighbor(std::size_t agentNo,
                                                std::size_t neighborNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->agentNeighbors_[neighborNo].second->id_;
  }
  return 0;
}

std::size_t RVOSimulator::getAgentMaxNeighbors(std::size_t agentNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->maxNeighbors_;
  }
  return 0;
}

float RVOSimulator::getAgentMaxSpeed(std::size_t agentNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->maxSpeed_;
  }
  return 0;
}

float RVOSimulator::getAgentNeighborDist(std::size_t agentNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->neighborDist_;
  }
  return 0;
}

std::size_t RVOSimulator::getAgentNumAgentNeighbors(std::size_t agentNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->agentNeighbors_.size();
  }
  return 0;
}

std::size_t RVOSimulator::getAgentNumObstacleNeighbors(
    std::size_t agentNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->obstacleNeighbors_.size();
  }
  return 0;
}

std::size_t RVOSimulator::getAgentNumORCALines(std::size_t agentNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->orcaLines_.size();
  }
  return 0;
}

std::size_t RVOSimulator::getAgentObstacleNeighbor(
    std::size_t agentNo, std::size_t neighborNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->obstacleNeighbors_[neighborNo].second->id_;
  }
  return 0;
}

const Line &RVOSimulator::getAgentORCALine(std::size_t agentNo,
                                           std::size_t lineNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->orcaLines_[lineNo];
  }
  return Line();
}

const Vector2 &RVOSimulator::getAgentPosition(std::size_t agentNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->position_;
  }
  return g_InValidVec2;
}

const Vector2 &RVOSimulator::getAgentPrefVelocity(std::size_t agentNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->prefVelocity_;
  }
  return g_InValidVec2;
}

float RVOSimulator::getAgentRadius(std::size_t agentNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->radius_;
  }
  return 0.0F;
}

float RVOSimulator::getAgentTimeHorizon(std::size_t agentNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->timeHorizon_;
  }
  return 0.0F;
}

float RVOSimulator::getAgentTimeHorizonObst(std::size_t agentNo) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->timeHorizonObst_;
  }
  return 0.0F;
}

const Vector2 &RVOSimulator::getAgentVelocity(std::size_t agentNo, bool & bIsVaild) const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    return it->second->velocity_;
  }
  bIsVaild = true;
  return g_InValidVec2;
}

const Vector2 &RVOSimulator::getObstacleVertex(std::size_t vertexNo) const {
  return obstacles_[vertexNo]->point_;
}

std::size_t RVOSimulator::getNextObstacleVertexNo(std::size_t vertexNo) const {
  return obstacles_[vertexNo]->next_->id_;
}

std::size_t RVOSimulator::getPrevObstacleVertexNo(std::size_t vertexNo) const {
  return obstacles_[vertexNo]->previous_->id_;
}

void RVOSimulator::processObstacles() { kdTree_->buildObstacleTree(); }

bool RVOSimulator::queryVisibility(const Vector2 &point1,
                                   const Vector2 &point2) const {
  return kdTree_->queryVisibility(point1, point2, 0.0F);
}

bool RVOSimulator::queryVisibility(const Vector2 &point1, const Vector2 &point2,
                                   float radius) const {
  return kdTree_->queryVisibility(point1, point2, radius);
}

void RVOSimulator::setAgentDefaults(float neighborDist,
                                    std::size_t maxNeighbors, float timeHorizon,
                                    float timeHorizonObst, float radius,
                                    float maxSpeed) {
  setAgentDefaults(neighborDist, maxNeighbors, timeHorizon, timeHorizonObst,
                   radius, maxSpeed, Vector2());
}

void RVOSimulator::setAgentDefaults(float neighborDist,
                                    std::size_t maxNeighbors, float timeHorizon,
                                    float timeHorizonObst, float radius,
                                    float maxSpeed, const Vector2 &velocity) {
  if (defaultAgent_ == NULL) {
    defaultAgent_ = new Agent();
  }

  defaultAgent_->maxNeighbors_ = maxNeighbors;
  defaultAgent_->maxSpeed_ = maxSpeed;
  defaultAgent_->neighborDist_ = neighborDist;
  defaultAgent_->radius_ = radius;
  defaultAgent_->timeHorizon_ = timeHorizon;
  defaultAgent_->timeHorizonObst_ = timeHorizonObst;
  defaultAgent_->velocity_ = velocity;
}

void RVOSimulator::setAgentMaxNeighbors(std::size_t agentNo,
                                        std::size_t maxNeighbors) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    it->second->maxNeighbors_ = maxNeighbors;
  }

}

void RVOSimulator::setAgentMaxSpeed(std::size_t agentNo, float maxSpeed) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    it->second->maxSpeed_ = maxSpeed;
  }
}

void RVOSimulator::setAgentNeighborDist(std::size_t agentNo,
                                        float neighborDist) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    it->second->neighborDist_ = neighborDist;
  }
}

void RVOSimulator::setAgentPosition(std::size_t agentNo,
                                    const Vector2 &position) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    it->second->position_ = position;
  }
}

void RVOSimulator::setAgentPrefVelocity(std::size_t agentNo,
                                        const Vector2 &prefVelocity) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    it->second->prefVelocity_ = prefVelocity;
  }
}

void RVOSimulator::setAgentRadius(std::size_t agentNo, float radius) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    it->second->radius_ = radius;
  }
}

void RVOSimulator::setAgentTimeHorizon(std::size_t agentNo, float timeHorizon) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    it->second->timeHorizon_ = timeHorizon;
  }
}

void RVOSimulator::setAgentTimeHorizonObst(std::size_t agentNo,
                                           float timeHorizonObst) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    it->second->timeHorizonObst_ = timeHorizonObst;
  }
}

void RVOSimulator::setAgentVelocity(std::size_t agentNo,
                                    const Vector2 &velocity) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    it->second->velocity_ = velocity;
  }
}

void RVOSimulator::setIsMoving(std::size_t agentNo, bool isMoving) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = agents_.find(agentNo);
  if (it != agents_.end()) {
    it->second->isMoving_ = isMoving;
    }
}
std::size_t RVOSimulator::getNumAgents() const {
  std::lock_guard<std::mutex> lock(agentMutex_);
  return agents_.size();
}

} /* namespace RVO */
