// Repository: d-gamedev-team/dbox
// File: examples/demo/tests/continuoustest.d

/*
 * Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgment in the product documentation would be
 * appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */
module tests.continuoustest;

import core.stdc.math;

import std.string;
import std.typecons;

import deimos.glfw.glfw3;

import dbox;

import framework.debug_draw;
import framework.test;

class ContinuousTest : Test
{
    this()
    {
        {
            b2BodyDef bd;
            bd.position.Set(0.0f, 0.0f);
            b2Body* body_ = m_world.CreateBody(&bd);

            b2EdgeShape edge = new b2EdgeShape();

            edge.Set(b2Vec2(-10.0f, 0.0f), b2Vec2(10.0f, 0.0f));
            body_.CreateFixture(edge, 0.0f);

            auto shape = new b2PolygonShape();
            shape.SetAsBox(0.2f, 1.0f, b2Vec2(0.5f, 1.0f), 0.0f);
            body_.CreateFixture(shape, 0.0f);
        }

        {
            b2BodyDef bd;
            bd.type = b2_dynamicBody;
            bd.position.Set(0.0f, 20.0f);

            // bd.angle = 0.1f;

            auto shape = new b2PolygonShape();
            shape.SetAsBox(2.0f, 0.1f);

            m_body = m_world.CreateBody(&bd);
            m_body.CreateFixture(shape, 1.0f);

            m_angularVelocity = RandomFloat(-50.0f, 50.0f);

            // m_angularVelocity = 46.661274f;
            m_body.SetLinearVelocity(b2Vec2(0.0f, -100.0f));
            m_body.SetAngularVelocity(m_angularVelocity);
        }

        b2_gjkCalls        = 0;
        b2_gjkIters        = 0;
        b2_gjkMaxIters     = 0;
        b2_toiCalls        = 0;
        b2_toiIters        = 0;
        b2_toiRootIters    = 0;
        b2_toiMaxRootIters = 0;
        b2_toiTime         = 0.0f;
        b2_toiMaxTime      = 0.0f;
    }

    override void Keyboard(int key)
    {
        switch (key)
        {
            case GLFW_KEY_W:
            {
                b2Vec2 f = m_body.GetWorldVector(b2Vec2(0.0f, -200.0f));
                b2Vec2 p = m_body.GetWorldPoint(b2Vec2(0.0f, 2.0f));
                m_body.ApplyForce(f, p, true);
            }
            break;

            case GLFW_KEY_A:
            {
                m_body.ApplyTorque(50.0f, true);
            }
            break;

            case GLFW_KEY_D:
            {
                m_body.ApplyTorque(-50.0f, true);
            }
            break;

            default:
                break;
        }
    }

    void Launch()
    {
        b2_gjkCalls        = 0;
        b2_gjkIters        = 0;
        b2_gjkMaxIters     = 0;
        b2_toiCalls        = 0;
        b2_toiIters        = 0;
        b2_toiRootIters    = 0;
        b2_toiMaxRootIters = 0;
        b2_toiTime         = 0.0f;
        b2_toiMaxTime      = 0.0f;

        m_body.SetTransform(b2Vec2(0.0f, 20.0f), 0.0f);
        m_angularVelocity = RandomFloat(-50.0f, 50.0f);
        m_body.SetLinearVelocity(b2Vec2(0.0f, -100.0f));
        m_body.SetAngularVelocity(m_angularVelocity);
    }

    override void Step(Settings* settings)
    {
        super.Step(settings);

        if (b2_gjkCalls > 0)
        {
            g_debugDraw.DrawString(5, m_textLine, format("gjk calls = %d, ave gjk iters = %3.1f, max gjk iters = %d",
                                                         b2_gjkCalls, b2_gjkIters / cast(float32)b2_gjkCalls, b2_gjkMaxIters));
            m_textLine += DRAW_STRING_NEW_LINE;
        }

        if (b2_toiCalls > 0)
        {
            g_debugDraw.DrawString(5, m_textLine, format("toi calls = %d, ave [max] toi iters = %3.1f [%d]",
                                                         b2_toiCalls, b2_toiIters / cast(float32)b2_toiCalls, b2_toiMaxRootIters));
            m_textLine += DRAW_STRING_NEW_LINE;

            g_debugDraw.DrawString(5, m_textLine, format("ave [max] toi root iters = %3.1f [%d]",
                                                         b2_toiRootIters / cast(float32)b2_toiCalls, b2_toiMaxRootIters));
            m_textLine += DRAW_STRING_NEW_LINE;

            g_debugDraw.DrawString(5, m_textLine, format("ave [max] toi time = %.1f [%.1f] (microseconds)",
                                                         1000.0f * b2_toiTime / cast(float32)b2_toiCalls, 1000.0f * b2_toiMaxTime));
            m_textLine += DRAW_STRING_NEW_LINE;
        }

        if (m_stepCount % 60 == 0)
        {
            Launch();
        }
    }

    b2Body* m_body;
    float32 m_angularVelocity;

    static Test Create()
    {
        return new typeof(this);
    }
}
