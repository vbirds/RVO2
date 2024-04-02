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

RVOSimulator::RVOSimulator()
    : defaultAgent_(NULL),
      kdTree_(new KdTree(this)),
      globalTime_(0.0F),
      timeStep_(0.0F),
      totalID_(0) {}

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
  delete defaultAgent_;
  delete kdTree_;

  for (std::size_t i = 0U; i < agents_.size(); ++i) {
    delete agents_[i];
  }

  for (std::size_t i = 0U; i < obstacles_.size(); ++i) {
    delete obstacles_[i];
  }
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
    agents_.push_back(agent);
    onAddAgent();
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
  agents_.push_back(agent);
  onAddAgent();
  return agent->id_;
}

void RVOSimulator::delAgetent(std::size_t agentNo) {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    agents_[it->second]->needDelete_ = true;
  }
}

void RVOSimulator::updateDeleteAgent() {
  bool isDelete = false;
  for (std::vector<Agent *>::iterator it = agents_.begin(); it != agents_.end(); ++it) {
    if ((*it)->needDelete_) {
      delete *it;
      agents_.erase(it);
      --it;;
      isDelete = true;
    }
  }
  if (isDelete) {
    onDelAgent();
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
  updateDeleteAgent();
  kdTree_->buildAgentTree();

#ifdef _OPENMP
#pragma omp parallel for
#endif /* _OPENMP */
  for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
    agents_[i]->computeNeighbors(kdTree_);
    agents_[i]->computeNewVelocity(timeStep_);
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif /* _OPENMP */
  for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
    agents_[i]->update(timeStep_);
  }

  globalTime_ += timeStep_;
}

std::size_t RVOSimulator::getAgentAgentNeighbor(std::size_t agentNo,
                                                std::size_t neighborNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->agentNeighbors_[neighborNo].second->id_;
  }
  return 0;
}

std::size_t RVOSimulator::getAgentMaxNeighbors(std::size_t agentNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->maxNeighbors_;
  }
  return 0;
}

float RVOSimulator::getAgentMaxSpeed(std::size_t agentNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->maxSpeed_;
  }
  return 0;
}

float RVOSimulator::getAgentNeighborDist(std::size_t agentNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->neighborDist_;
  }
  return 0;
}

std::size_t RVOSimulator::getAgentNumAgentNeighbors(std::size_t agentNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->agentNeighbors_.size();
  }
  return 0;
}

std::size_t RVOSimulator::getAgentNumObstacleNeighbors(
    std::size_t agentNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->obstacleNeighbors_.size();
  }
  return 0;
}

std::size_t RVOSimulator::getAgentNumORCALines(std::size_t agentNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->orcaLines_.size();
  }
  return 0;
}

std::size_t RVOSimulator::getAgentObstacleNeighbor(
    std::size_t agentNo, std::size_t neighborNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->obstacleNeighbors_[neighborNo].second->id_;
  }
  return 0;
}

const Line &RVOSimulator::getAgentORCALine(std::size_t agentNo,
                                           std::size_t lineNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->orcaLines_[lineNo];
  }
  return Line();
}

const Vector2 &RVOSimulator::getAgentPosition(std::size_t agentNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->position_;
  }
  return Vector2();
}

const Vector2 &RVOSimulator::getAgentPrefVelocity(std::size_t agentNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->prefVelocity_;
  }
  return Vector2();
}

float RVOSimulator::getAgentRadius(std::size_t agentNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->radius_;
  }
  return 0.0F;
}

float RVOSimulator::getAgentTimeHorizon(std::size_t agentNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->timeHorizon_;
  }
  return 0.0F;
}

float RVOSimulator::getAgentTimeHorizonObst(std::size_t agentNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->timeHorizonObst_;
  }
  return 0.0F;
}

const Vector2 &RVOSimulator::getAgentVelocity(std::size_t agentNo) const {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    return agents_[it->second]->velocity_;
  }
  return Vector2();
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
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    agents_[it->second]->maxNeighbors_ = maxNeighbors;
  }

}

void RVOSimulator::setAgentMaxSpeed(std::size_t agentNo, float maxSpeed) {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    agents_[it->second]->maxSpeed_ = maxSpeed;
  }
}

void RVOSimulator::setAgentNeighborDist(std::size_t agentNo,
                                        float neighborDist) {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    agents_[it->second]->neighborDist_ = neighborDist;
  }
}

void RVOSimulator::setAgentPosition(std::size_t agentNo,
                                    const Vector2 &position) {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    agents_[it->second]->position_ = position;
  }
}

void RVOSimulator::setAgentPrefVelocity(std::size_t agentNo,
                                        const Vector2 &prefVelocity) {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    agents_[it->second]->prefVelocity_ = prefVelocity;
  }
}

void RVOSimulator::setAgentRadius(std::size_t agentNo, float radius) {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    agents_[it->second]->radius_ = radius;
  }
}

void RVOSimulator::setAgentTimeHorizon(std::size_t agentNo, float timeHorizon) {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    agents_[it->second]->timeHorizon_ = timeHorizon;
  }
}

void RVOSimulator::setAgentTimeHorizonObst(std::size_t agentNo,
                                           float timeHorizonObst) {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    agents_[it->second]->timeHorizonObst_ = timeHorizonObst;
  }
}

void RVOSimulator::setAgentVelocity(std::size_t agentNo,
                                    const Vector2 &velocity) {
  std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
  if (it != agentNo2indexDict_.end()) {
    agents_[it->second]->velocity_ = velocity;
  }
}

void RVOSimulator::onAddAgent() {
  if (agents_.empty()) {
    return;
  }
  int index = static_cast<int>(agents_.size() - 1);
  std::size_t agentNo = agents_[index]->id_;
  agentNo2indexDict_.insert(std::make_pair(agentNo, index));
  index2agentNoDict_.insert(std::make_pair(index, agentNo));
}

void RVOSimulator::onDelAgent() {
  agentNo2indexDict_.clear();
  index2agentNoDict_.clear();

  for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
    std::size_t agentNo = agents_[i]->id_;
    agentNo2indexDict_.insert(std::make_pair(agentNo, i));
    index2agentNoDict_.insert(std::make_pair(i, agentNo));
  }

}

void RVOSimulator::setIsMoving(std::size_t agentNo, bool isMoving) {
    std::unordered_map<std::size_t, int>::const_iterator it = agentNo2indexDict_.find(agentNo);
    if (it != agentNo2indexDict_.end()) {
      agents_[it->second]->isMoving_ = isMoving;
    }
}

} /* namespace RVO */
